from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
from sklearn.metrics import accuracy_score
import json
import numpy as np


TRAIN_DATA_FILEPATH = ".local/bird_testing_set.json"
MODEL_NAME = "answerdotai/ModernBERT-large"


def _load_filtered_dataset(filepath):
    with open(filepath, "r") as fp:
        raw_data = json.load(fp)

    filtered_data = [
        {
            "question": entry["question"],
            "schema": entry["schema"],
            "label": int(entry["feasible"])
        }
        for entry in raw_data
        if entry.get("feasible") is not None
    ]
    return Dataset.from_list(filtered_data)


def tokenize_function(batch, tokenizer):
    combined_inputs = [f"{q} [SEP] {s}" for q, s in zip(batch["question"], batch["schema"])]
    return tokenizer(combined_inputs, truncation=True, padding="max_length")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds)
    }


def train():
    dataset = _load_filtered_dataset(TRAIN_DATA_FILEPATH)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    training_args = TrainingArguments(
        output_dir="./modernbert-classifier",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        save_total_limit=2,
        evaluation_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=32,  # H100s can easily handle this
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=1,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=True,  # Mixed precision
        report_to="none",  # Disable W&B or TensorBoard if not needed
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    trainer.train()


if __name__ == "__main__":
    train()
