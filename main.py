from src.core.model_implementations import LlamaModel, DeepSeekLlamaModel, DeepSeekQwenModel, XiYanSQLModel, ModelWithSQLCoderAbstentionModule
from src.core.prompt_strategies import Llama3PromptStrategy, DeepSeekPromptStrategy, XiYanSQLPromptStrategy, SQLCoderAbstentionPromptStrategy
from src.database.database import verify_database, get_conn
from src.core.base_model import NL2SQLModel
from src.common.logger import get_logger
from src.common.reporting import Reporter
from json import load, dump
from datetime import date
import os
from dotenv import load_dotenv

logger = get_logger(__name__)
load_dotenv()

SQL_DIALECT = os.getenv('SQL_DIALECT')
DATASET_PATH = os.getenv('DATASET_PATH')
RESULTS_DIR = os.getenv('RESULTS_DIR')
DB_NAME = os.getenv('DB_NAME')
DATASET_NAME = os.getenv('DATASET_NAME')
MODEL = os.getenv('MODEL')
NUMBER_OF_RUNS = int(os.getenv('NUMBER_OF_RUNS', 1))
DB_NATURAL = bool(int(os.getenv('DB_NATURAL', 0)))
MSCHEMA = bool(int(os.getenv('MSCHEMA', 1)))
WITH_SGAM = bool(int(os.getenv('WITH_SGAM', 0)))
DATE = date.today()


def load_dataset(dataset_path: str):
    with open(dataset_path, "r") as file:
        dataset = load(file)
    if WITH_SGAM:
        return dataset

    return [{"question": pair["question"], "golden_query": pair["goal_query"]} for pair in dataset]


def save_results(results_path: str, model: NL2SQLModel) -> None:
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as file:
        dump(model.results, file, indent=4)


def get_model() -> NL2SQLModel:
    dataset = load_dataset(DATASET_PATH)

    match MODEL:
        case "XiYan":
            prompt_strategy = XiYanSQLPromptStrategy(SQL_DIALECT)
            model = XiYanSQLModel(dataset, prompt_strategy, MSCHEMA)
        case "DeepSeekQwen":
            prompt_strategy = DeepSeekPromptStrategy(SQL_DIALECT)
            model = DeepSeekQwenModel(dataset, prompt_strategy, MSCHEMA)
        case "Llama":
            prompt_strategy = Llama3PromptStrategy(SQL_DIALECT)
            model = LlamaModel(dataset, prompt_strategy, MSCHEMA)
        case "DeepSeekLlama":
            prompt_strategy = DeepSeekPromptStrategy(SQL_DIALECT)
            model = DeepSeekLlamaModel(dataset, prompt_strategy, MSCHEMA)

    return model


def run_experiments(model: NL2SQLModel) -> None:
    for i in range(NUMBER_OF_RUNS):
        model.run(WITH_SGAM)
        file_name = f"{MODEL}{DATASET_NAME}{'Natural' if DB_NATURAL else 'Abbreviated'}{'MSchema' if MSCHEMA else ''}{'SGAM' if WITH_SGAM else ''}_{i + 1}.json"
        save_results(f"{RESULTS_DIR}/{DB_NAME}/{MODEL}/{'Natural' if DB_NATURAL else 'Abbreviated'}/{DATE}/{file_name}", model)
        model.results = {}


if __name__ == "__main__":
    if not verify_database(get_conn()):
        raise RuntimeError("Database was malformed, check log for details.")
    model = get_model()
    reporter = Reporter()
    result_directory = f"{RESULTS_DIR}/{DB_NAME}/{MODEL}/{'Natural' if DB_NATURAL else 'Abbreviated'}/{DATE}"

    run_experiments(model)
    reporter.generate_report(result_directory)
