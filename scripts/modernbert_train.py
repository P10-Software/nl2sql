import optuna
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from src.common.logger import get_logger
from datasets import Dataset
from sklearn.metrics import precision_score, recall_score
import json
import numpy as np


logger = get_logger()


TRAIN_DATA_FILEPATH = ".local/bird_testing_set.json"
MODEL_NAME = "answerdotai/ModernBERT-large"


def objective(trial: optuna.Trial):
    """
    Defines the training objective for an Optuna trial.
    Sets and applies sampled hyperparameters, trains a model,
    and returns a skewed F-beta score for evaluation.
    """
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 5e-5, log=True)
    num_train_epochs = trial.suggest_int("num_train_epochs", 2, 5)
    logger.info(f"Running trial {trial.number}, using parameters: {trial.params}")

    dataset = _load_filtered_dataset(TRAIN_DATA_FILEPATH)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized_dataset = dataset.map(
        lambda x: _tokenize_function(x, tokenizer), batched=True)
    tokenized_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"])

    training_args = TrainingArguments(
        output_dir=f".local/optuna-bert-{trial.number}",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        learning_rate=learning_rate,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        fp16=True,
    )

    trainer = Trainer(
        model_init=_model_init,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=_compute_metrics_custom,
        tokenizer=tokenizer,
    )

    trainer.train()

    eval_results = trainer.evaluate()

    return eval_results["skewed_fbeta"]


def run_study():
    """
    Creates or resumes an Optuna study and launches hyperparameter 
    optimization.
    """
    study = optuna.create_study(
        direction="maximize",
        study_name="modernbert_study",
        storage="sqlite:///optuna_study.db",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=50)
    logger.info(f"Best trial: {study.best_trial}")


def _tokenize_function(batch, tokenizer):
    combined_inputs = [
        f"{q} [SEP] {s}" for q, s in zip(batch["question"], batch["schema"])]
    return tokenizer(combined_inputs, truncation=True, padding="max_length")


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


def _compute_fbeta(preds, labels, b=4):
    """
    Computes a fβ measure, defaults to f4.
    """
    precision = precision_score(y_true=labels, y_pred=preds)
    recall = recall_score(y_true=labels, y_pred=preds)

    skewed_fb = (1 + b**2) * (precision * recall) / ((b**2 * precision) + recall)
    return skewed_fb


def _compute_metrics_custom(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "skewed_fbeta": _compute_fbeta(preds, labels),
    }


def _model_init(trial):
    return AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)


if __name__ == "__main__":
    run_study()
