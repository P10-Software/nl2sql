from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback)
from datasets import load_dataset
from src.common.logger import get_logger
from scipy.stats import entropy
import numpy as np
import torch

logger = get_logger(__name__)
SAVE_LOCALE = ".local/trust_sql"
DATASET_LOCALE = ""
SQL_CODER_MODEL = "defog/sqlcoder-7b-2"


def train_t5_sql_gen():
    dataset = load_dataset(DATASET_LOCALE)

    tokenizer = T5Tokenizer.from_pretrained("t5-3b")
    model = T5ForConditionalGeneration.from_pretrained("t5-3b")

    def preprocess(example):
        input_text = f"{example.get('question')} {example.get('schema')}"
        target_text = example.get('sql')
        return tokenizer(
            input_text,
            text_target=target_text,
            truncation=True,
            padding="max_length",
            max_length=512)

    tokenized_dataset = dataset.map(preprocess)
    train_dataset = tokenized_dataset.train_test_split(test_size=0.1)

    args = Seq2SeqTrainingArguments(
        output_dir=f"{SAVE_LOCALE}/t5_sqlgen",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=16,
        bf16=True,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        num_train_epochs=10
    )

    early_stopping = EarlyStoppingCallback(early_stopping_patience=2)

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset["train"],
        eval_dataset=train_dataset["eval"],
        processing_class=tokenizer,
        callbacks=early_stopping
    )

    trainer.train()

    logits_list = []
    labels = []
    model.eval()

    for example in train_dataset['eval']:
        input_text = f"{example['question']} {example['schema']}"
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

        output = model.generate(**inputs)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        logits_list.append(logits)

        generated_sql = tokenizer.decode(output[0], skip_special_tokens=True)

        correct = 1 if generated_sql.strip() == example['sql'].strip() else 0
        labels.append(correct)

    maxent_threshold = find_t5_maxent_threshold(logits_list, labels)

    model.config.maxent_threshold = maxent_threshold

    trainer.save_model(f"{SAVE_LOCALE}/t5_sqlgen/final")
    tokenizer.save_pretrained(f"{SAVE_LOCALE}/t5_sqlgen/tokenizer")


def train_sqlcoder_feasibility():
    dataset = load_dataset(DATASET_LOCALE)

    tokenizer = AutoTokenizer(SQL_CODER_MODEL)
    model = AutoModelForSequenceClassification(SQL_CODER_MODEL)
    context_size = getattr(model.model.config, "max_position_embeddings", None)

    def preprocess(example):
        input_text = f"{example.get('question')} {example.get('schema')}"
        label = int(example.get('infeasible', 0))
        tokenized = tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=context_size)
        tokenized['label'] = label
        return tokenized

    tokenized_dataset = dataset.map(preprocess)
    train_dataset = tokenized_dataset.train_test_split(test_size=0.1)

    args = TrainingArguments(
        output_dir=f"{SAVE_LOCALE}/sql_coder_infeasible",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=16,
        bf16=True,
        num_train_epochs=1,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset['train'],
        eval_dataset=train_dataset['eval'],
        processing_class=tokenizer
    )

    trainer.train()
    trainer.save_model(f"{SAVE_LOCALE}/sql_coder_infeasible/final")
    tokenizer.save_pretrained(f"{SAVE_LOCALE}/sql_coder_infeasible/tokenizer")


def train_sqlcoder_error_detect():
    dataset = load_dataset(DATASET_LOCALE)

    tokenizer = AutoTokenizer(SQL_CODER_MODEL)
    model = AutoModelForSequenceClassification(SQL_CODER_MODEL)
    context_size = getattr(model.model.config, "max_position_embeddings", None)

    def preprocess(example):
        input_text = f"{example.get('question')} {example.get('sql')}"
        label = int(example.get("infeasible", 0))
        tokenized = tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=context_size)
        tokenized['label'] = label
        return tokenized

    tokenized_dataset = dataset.map(preprocess)
    train_dataset = tokenized_dataset.train_test_split(test_size=0.1)

    args = TrainingArguments(
        output_dir=f"{SAVE_LOCALE}/sql_coder_error",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=16,
        bf16=True,
        num_train_epochs=1,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset['train'],
        eval_dataset=train_dataset['eval'],
        processing_class=tokenizer
    )

    trainer.train()
    trainer.save_model(f"{SAVE_LOCALE}/sql_coder_error/final")
    tokenizer.save_pretrained(f"{SAVE_LOCALE}/sql_coder_error/tokenizer")


def find_t5_maxent_threshold(logits_list, labels):
    def compute_entropy(probs):
        return entropy(probs, base=2)

    entropies = []
    for logits in logits_list:
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        avg_entropy = np.mean([compute_entropy(p) for p in probs])
        entropies.append(avg_entropy)

    sample_scores = list(zip(entropies, [1 if is_correct else -1 for is_correct in labels]))

    sorted_scores = sorted(sample_scores, key=lambda x: x[0])

    cumulative = np.cumsum([s for _, s in sorted_scores])
    best_idx = int(np.argmax(cumulative))
    best_threshold = sorted_scores[best_idx][0]

    return best_threshold


if __name__ == "__main__":
    logger.info("Starting training of models for TrustSQL...")
    logger.info("... Training T5 for SQL generation...")
    train_t5_sql_gen()
    logger.info("... Training SQLCoder-7b-2 for feasibility classification...")
    train_sqlcoder_feasibility()
    logger.info("... Training SQLCoder-7b-2 for SQL error classification...")
    train_sqlcoder_error_detect()
    logger.info("...Finished training all models...")
    logger.info("Finished all tasks for training TrustSQL...")
    logger.info(f"Results saved to: {SAVE_LOCALE}")
