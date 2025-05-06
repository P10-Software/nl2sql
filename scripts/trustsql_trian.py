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

logger = get_logger(__name__)
SAVE_LOCALE = ""
DATASET_LOCALE = ""


def train_t5_sql_gen():
    dataset = load_dataset(DATASET_LOCALE)

    tokenizer = T5Tokenizer.from_pretrained("t5-3b")
    model = T5ForConditionalGeneration.from_pretrained("t5-3b")

    def preprocess(example):
        input_text = f"{example.get('question')} {example.get('schema')}"
        target_text = example.get('sql')
        return tokenizer(input_text, text_target=target_text, truncation=True, padding="max_length", max_length=512)

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
        greater_is_better=False
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


def train_sqlcoder_feasibility():
    raise NotImplementedError()


def train_sqlcoder_error_detect():
    raise NotImplementedError()


def find_t5_maxent_threshold():
    raise NotImplementedError()


if __name__ == "__main__":
    logger.info("Starting training of models for TrustSQL...")
    logger.info("... Training T5 for SQL generation...")
    train_t5_sql_gen()
    logger.info("... Training SQLCoder-7b-2 for feasibility classification...")
    train_sqlcoder_feasibility()
    logger.info("... Training SQLCoder-7b-2 for SQL error classification...")
    train_sqlcoder_error_detect()
    logger.info("...Finished training all models...")
    logger.info("...Calculating MAXENT threshold...")
    find_t5_maxent_threshold()
    logger.info("Finished all tasks for training TrustSQL...")
    logger.info(f"Results saved to: {SAVE_LOCALE}")
