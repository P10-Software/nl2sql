from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM)
from datasets import load_dataset
from src.common.logger import get_logger
import numpy as np
import torch
from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
import gc

logger = get_logger(__name__)
SAVE_LOCALE = ".local/trust_sql"
SQL_CODER_MODEL = "defog/sqlcoder-7b-2"


def train_t5_sql_gen():
    data_files = {"train": ".local/trust_sql/bird_train_set.json"}
    dataset = load_dataset('json', data_files=data_files)

    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-3b")
    model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-3b")

    def preprocess(example):
        start_prompt = "Database:\n"
        middle_prompt = "\n\nQuestion:\n"
        end_prompt = "\n\nAnswer:\n"
        data_zip = zip(example['schema'], example['question'])
        prompt = [start_prompt + context + middle_prompt + question + end_prompt for context, question in data_zip]

        model_inputs = {}

        model_inputs['input_ids'] = tokenizer(prompt, padding='max_length', truncation=True, return_tensors="pt", max_length=512).input_ids
        model_inputs['labels'] = tokenizer(example['SQL'], padding='max_length', truncation=True, return_tensors="pt", max_length=512).input_ids
        return model_inputs

    tokenized_dataset = dataset.map(preprocess, batched=True)
    train_dataset = tokenized_dataset["train"].train_test_split(test_size=0.1, shuffle=True)

    args = TrainingArguments(
        output_dir=f"{SAVE_LOCALE}/t5_sqlgen",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=1,
        bf16=True,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        num_train_epochs=10
    )

    early_stopping = EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.01)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset["train"],
        eval_dataset=train_dataset["test"],
        processing_class=tokenizer,
        callbacks=[early_stopping]
    )

    trainer.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    entropy_label_pairs = []

    model.eval()

    for example in train_dataset['test']:
        input_text = f"{example['question']} {example['schema']}"

        # Tokenize and move to device
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate output
        output = model.generate(**inputs)

        # Get logits for entropy thresholding or confidence
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * probs.log()).sum(dim=-1).mean().item()  # mean over sequence length

        # Decode and compare
        generated_sql = tokenizer.decode(output[0], skip_special_tokens=True)
        label = 1 if generated_sql.strip() == example['sql'].strip() else 0

        entropy_label_pairs.append((label, entropy))

    maxent_threshold = find_t5_maxent_threshold(entropy_label_pairs)

    model.config.maxent_threshold = maxent_threshold

    trainer.save_model(f"{SAVE_LOCALE}/t5_sqlgen/final")
    tokenizer.save_pretrained(f"{SAVE_LOCALE}/t5_sqlgen/tokenizer")


def train_sqlcoder_feasibility():
    data_files = {"train": ".local/trust_sql/bird_abstention_train_set.json"}
    dataset = load_dataset('json', data_files=data_files)

    tokenizer = AutoTokenizer.from_pretrained(SQL_CODER_MODEL)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        SQL_CODER_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model.resize_token_embeddings(len(tokenizer))

    model = get_peft_model(model, lora_config)

    def preprocess(example):
        input_text = f"""
            ### Task
            Generate a SQLite SQL query to answer [QUESTION]{example['question']}[/QUESTION]

            ### Instructions 
            - If you cannot answer the question with the available database schema, return 'I do not know'

            ### Database Schema
            The query will run on a database with the following schema:
            {example['schema']}

            ### Answer
            Given the database schema, here is the SQL query that [QUESTION]{example['question']}[/QUESTION]
            [SQL]
        """
        if example.get('feasible', 'yes') == 'no':
            label_text = "I do not know"
        else:
            label_text = "answerable"
        tokenized = tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=4096)
        labels = tokenizer(
            label_text,
            truncation=True,
            padding="max_length",
            max_length=4096
        )
        tokenized["labels"] = labels["input_ids"]
        return tokenized

    tokenized_dataset = dataset.map(preprocess, remove_columns=dataset['train'].column_names)
    train_test = tokenized_dataset["train"].train_test_split(test_size=0.1, shuffle=True)

    train_dataset = train_test["train"]
    eval_dataset = train_test["test"]

    args = TrainingArguments(
        output_dir=f"{SAVE_LOCALE}/sql_coder_infeasible",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
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
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer
    )

    trainer.train()
    trainer.save_model(f"{SAVE_LOCALE}/sql_coder_infeasible/final")
    tokenizer.save_pretrained(f"{SAVE_LOCALE}/sql_coder_infeasible/tokenizer")


def train_sqlcoder_error_detect():
    data_files = {"train": ".local/trust_sql/bird_abstention_train_set.json"}
    dataset = load_dataset('json', data_files=data_files)

    tokenizer = AutoTokenizer.from_pretrained(SQL_CODER_MODEL)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        SQL_CODER_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model.resize_token_embeddings(len(tokenizer))

    model = get_peft_model(model, lora_config)

    def preprocess(example):
        input_text = f"""
            #### Based on the question and predicted SQLite SQL, are you sure the SQL below is correct? If you consider the SQL is correct, answer me with 'correct'. 
            If not, answer me with 'incorrect'. Only output your response without explanation.

            ### Database Schema
            The query will run on a database with the following schema:
            {example['schema']}

            Question: {example['question']}
            Predicted SQL: {example['SQL']}
            Answer:        
        """
        if example.get('feasible', 'yes') == 'no':
            label_text = "incorrect"
        else:
            label_text = "correct"
        
        tokenized = tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=4096)
        labels = tokenizer(
            label_text,
            truncation=True,
            padding="max_length",
            max_length=4096
        )
        tokenized["labels"] = labels["input_ids"]
        return tokenized

    tokenized_dataset = dataset.map(preprocess)
    train_dataset = tokenized_dataset["train"].train_test_split(test_size=0.1, shuffle=True)

    args = TrainingArguments(
        output_dir=f"{SAVE_LOCALE}/sql_coder_error",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=2,
        bf16=True,
        num_train_epochs=1,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset['train'],
        eval_dataset=train_dataset['test'],
        processing_class=tokenizer,
        data_collator=data_collator
    )

    trainer.train()
    trainer.save_model(f"{SAVE_LOCALE}/sql_coder_error/final")
    tokenizer.save_pretrained(f"{SAVE_LOCALE}/sql_coder_error/tokenizer")


def find_t5_maxent_threshold(entropy_label_pairs):
    # for label, entropy in entropy_label_pairs:

    # sample_scores = list(zip(entropies, [1 if is_correct else -1 for is_correct in labels]))

    sorted_scores = sorted(entropy_label_pairs, key=lambda x: x[0])

    cumulative = np.cumsum([s for _, s in sorted_scores])
    best_idx = int(np.argmax(cumulative))
    best_threshold = sorted_scores[best_idx][0]

    return best_threshold


def cleanup_memory():
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    logger.info("Starting training of models for TrustSQL...")
    cleanup_memory()
    logger.info("... Training SQLCoder-7b-2 for feasibility classification...")
    train_sqlcoder_feasibility()
    cleanup_memory()

    logger.info("... Training SQLCoder-7b-2 for SQL error classification...")
    train_sqlcoder_error_detect()
    cleanup_memory()

    logger.info("... Training T5 for SQL generation...")
    train_t5_sql_gen()
    cleanup_memory()

    logger.info("...Finished training all models...")
    logger.info("Finished all tasks for training TrustSQL...")
    logger.info(f"Results saved to: {SAVE_LOCALE}")
