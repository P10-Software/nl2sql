from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments)
from datasets import load_dataset
from src.common.logger import get_logger

logger = get_logger(__name__)
SAVE_LOCALE = ""


def train_t5_sql_gen():
    raise NotImplementedError()


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
