from src.core.model_implementations import LlamaModel, DeepSeekLlamaModel, DeepSeekQwenModel, XiYanSQLModel, ModelWithAbstentionModule
from src.core.prompt_strategies import Llama3PromptStrategy, DeepSeekPromptStrategy, XiYanSQLPromptStrategy, SQLCoderAbstentionPromptStrategy
from src.database.setup_database import get_conn
from src.database.database import execute_query
from src.core.base_model import NL2SQLModel, translate_query_to_natural
from src.common.logger import get_logger
from src.common.reporting import Reporter
from json import load, dump
from datetime import date
import os

logger = get_logger(__name__)

SQL_DIALECT = "sqlite"
SCHEMA_SIZES = ["Full", "Tables", "Columns"]
DATASET_PATH = ".local/EX_sqlite.json"
RESULTS_DIR = "results"
NATURALNESS = "Normalized"
MODEL = "XiYan"
PRE_ABSTENTION = False
POST_ABSTENTION = False
MSCHEMA = False
DATE = date.today()

def load_dataset(dataset_path: str):
    with open(dataset_path, "r") as file:
        dataset = load(file)

    return [{"question": pair["question"], "golden_query": pair["goal_query"]} for pair in dataset]

def save_results(results_path: str, model: NL2SQLModel):
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as file:
        dump(model.results, file, indent=4)

if __name__ == "__main__":
    connection = get_conn()
    dataset = load_dataset(DATASET_PATH)

    match MODEL:
        case "XiYan":
            prompt_strategy = XiYanSQLPromptStrategy(SQL_DIALECT)
            model = XiYanSQLModel(connection, dataset, prompt_strategy, MSCHEMA)
        case "DeepSeekQwen":
            prompt_strategy = DeepSeekPromptStrategy(SQL_DIALECT)
            model = DeepSeekQwenModel(connection, dataset, prompt_strategy, MSCHEMA)
        case "Llama":
            prompt_strategy = Llama3PromptStrategy(SQL_DIALECT)
            model = LlamaModel(connection, dataset, prompt_strategy, MSCHEMA)
        case "DeepSeekLlama":
            prompt_strategy = DeepSeekPromptStrategy(SQL_DIALECT)
            model = DeepSeekLlamaModel(connection, dataset, prompt_strategy, MSCHEMA)

    if PRE_ABSTENTION or POST_ABSTENTION:
        abstention_prompt_strategy =  SQLCoderAbstentionPromptStrategy(SQL_DIALECT)
        model = ModelWithAbstentionModule(connection, dataset, abstention_prompt_strategy, model, PRE_ABSTENTION, POST_ABSTENTION, MSCHEMA)

    connection.close()

    # Run models and save generated queries
    for schema_size in SCHEMA_SIZES:
        model.run(schema_size, naturalness=True)
        file_name = f"{MODEL}{schema_size}{NATURALNESS}{"MSchema" if MSCHEMA else ""}{"PreAbstention" if PRE_ABSTENTION else ""}{"PostAbstention" if POST_ABSTENTION else ""}.json"
        save_results(f"{RESULTS_DIR}/{SQL_DIALECT}/{MODEL}/{NATURALNESS}/{DATE}/{file_name}", model)
        model.results = {}

    # Load results, execute queries and add to reporter
    reporter = Reporter()

    for result_file_name in os.listdir(f"{RESULTS_DIR}/{SQL_DIALECT}/{MODEL}/{NATURALNESS}/{DATE}/"):
        path = f"{RESULTS_DIR}/{SQL_DIALECT}/{MODEL}/{NATURALNESS}/{DATE}/{result_file_name}"

        if result_file_name == "report.html": continue
        
        with open(path, "r") as file_pointer:
            results = load(file_pointer)

        logger.info(f"Running results of database for {path}.")
        for res in results.values():
            if res['golden_query']:
                if NATURALNESS == "Normalized":
                    res['golden_query'] = translate_query_to_natural(res['golden_query'])

                res['golden_result'] = execute_query(res['golden_query'])
            else:
                res['golden_result'] = None
            
            if res['generated_query']:
                res['generated_result'] = execute_query(res['generated_query'])
            else:
                res['generated_result'] = None

        logger.info(f"Executed all queries on the database for {path}.")
    
        reporter.add_result(results, result_file_name.split('.')[0])

    reporter.create_report(f"{RESULTS_DIR}/{SQL_DIALECT}/{MODEL}/{NATURALNESS}/{DATE}")
