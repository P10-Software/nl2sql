from src.core.model_implementations import LlamaModel, DeepSeekLlamaModel, DeepSeekQwenModel, XiYanSQLModel
from src.core.prompt_strategies import Llama3PromptStrategy, DeepSeekPromptStrategy, XiYanSQLPromptStrategy
from src.database.setup_database import get_conn
from json import load

SQL_DIALECT = "postgres"
SCHEMA_SIZE = "full"
DATASET_PATH = "EX.json"

def load_dataset(dataset_path: str):
    with open(dataset_path, "r") as file:
        dataset = load(file)

    return [{"question": pair["question"], "golden_query": pair["goal_query"]} for pair in dataset]

if __name__ == "__main__":
    connection = get_conn()
    dataset = load_dataset(DATASET_PATH)
    prompt_strategy = XiYanSQLPromptStrategy(SQL_DIALECT)
    model = XiYanSQLModel(connection, dataset, prompt_strategy)
    model.run(SCHEMA_SIZE)
    
    # TODO: Do analysis and reporting