from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from evaluate import load
from src.common.logger import get_logger
import numpy as np
import optuna


MODEL = "answerdotai/ModernBERT-large"
RUN = 4
logger = get_logger(__name__)


def objective(trial):
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)

    data_files = {"train": ".local/bird_abstention_test_set.csv"}
    dataset = load_dataset("csv", data_files=data_files)

    def tokenize_function(example):
        return tokenizer(example["question"], example["schema"], padding="max_length", truncation=True, max_length=8192)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_data = tokenized_dataset["train"].train_test_split(test_size=0.1, shuffle=True)

    labels = dataset["train"]["label"]
    class_weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
    #for param in model.base_model.parameters():
    #    param.requires_grad = False
#
    #for param in model.classifier.parameters():
    #    param.requires_grad = True

    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True)
    num_epochs = trial.suggest_int('num_train_epochs', 2, 5)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

    training_args = TrainingArguments(
        output_dir=f".local/AbstentionClassifier/parameter_tuning_{RUN}/optuna_trial_{trial.number}",
        eval_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        save_total_limit=2,
        logging_dir=".local/AbstentionClassifier/logs",
        logging_steps=100,
        fp16=True
    )

    f1_metric = load("f1")
    precision_metric = load("precision")
    recall_metric = load("recall")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        f1 = f1_metric.compute(predictions=predictions, references=labels, pos_label=0)["f1"]
        precision = precision_metric.compute(predictions=predictions, references=labels, pos_label=0)["precision"]
        recall = recall_metric.compute(predictions=predictions, references=labels, pos_label=0)["recall"]

        logger.info("\n" + classification_report(labels, predictions, digits=4))

        return {
            "f1": f1,
            "precision": precision,
            "recall": recall
        }

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset = tokenized_data["train"],
        eval_dataset = tokenized_data["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()

    eval_results = trainer.evaluate()

    return eval_results["eval_f1"]

study = optuna.create_study(
    direction='maximize',
    study_name=f'abstention_classifier_tuning_{RUN}',
    storage='sqlite:///optuna_study.db',
    load_if_exists=True
    )
study.optimize(objective, n_trials=10)

logger.info("Best trial:")
trial = study.best_trial
logger.info(f"  Value: {trial.value}")
logger.info(f"  Params: {trial.params}")