from src.core.model_implementations import LlamaModel, DeepSeekLlamaModel, DeepSeekQwenModel, XiYanSQLModel
from src.core.prompt_strategies import Llama3PromptStrategy, DeepSeekPromptStrategy, XiYanSQLPromptStrategy
from src.database.setup_database import get_conn
from src.core.base_model import NL2SQLModel
from json import load, dump

SQL_DIALECT = "postgres"
SCHEMA_SIZE = "full"
DATASET_PATH = "EX.json"
RESULTS_PATH = "XiYanFullAbbreviated.json"

def load_dataset(dataset_path: str):
    with open(dataset_path, "r") as file:
        dataset = load(file)

    return [{"question": pair["question"], "golden_query": pair["goal_query"]} for pair in dataset]

def save_results(results_path: str, model: NL2SQLModel):
    with open(results_path, "w") as file:
        dump(model.results, file, indent=4)

if __name__ == "__main__":
    connection = get_conn()
    dataset = load_dataset(DATASET_PATH)
    prompt_strategy = XiYanSQLPromptStrategy(SQL_DIALECT)
    model = XiYanSQLModel(connection, dataset, prompt_strategy)
    model.run(SCHEMA_SIZE)
    save_results(RESULTS_PATH, model)
    
    # TODO: Do analysis and reporting