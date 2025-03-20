from src.core.model_implementations import LlamaModel, DeepSeekLlamaModel, DeepSeekQwenModel, XiYanSQLModel, ModelWithAbstentionModule
from src.core.prompt_strategies import Llama3PromptStrategy, DeepSeekPromptStrategy, XiYanSQLPromptStrategy, SQLCoderAbstentionPromptStrategy
from src.database.database import execute_query
from src.core.base_model import NL2SQLModel, translate_query_to_natural
from src.common.logger import get_logger
from src.common.reporting import Reporter
from json import load, dump
from datetime import date
import os
from dotenv import load_dotenv

logger = get_logger(__name__)
load_dotenv()

SQL_DIALECT = os.getenv('SQL_DIALECT')
SCHEMA_SIZES = ["Full", "Tables", "Columns"]
DATASET_PATH = os.getenv('DATASET_PATH')
RESULTS_DIR = os.getenv('RESULTS_DIR')
DB_NATURAL = int(os.getenv('DB_NATURAL'))
MODEL = os.getenv('MODEL')
PRE_ABSTENTION = int(os.getenv('PRE_ABSTENTION'))
POST_ABSTENTION = int(os.getenv('POST_ABSTENTION'))
MSCHEMA = int(os.getenv('MSCHEMA'))
DATE = date.today()

def load_dataset(dataset_path: str):
    with open(dataset_path, "r") as file:
        dataset = load(file)

    return [{"question": pair["question"], "golden_query": pair["goal_query"]} for pair in dataset]

def save_results(results_path: str, model: NL2SQLModel):
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as file:
        dump(model.results, file, indent=4)

def get_model():
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

    return model    

def run_experiments(model: NL2SQLModel):
    for schema_size in SCHEMA_SIZES:
        model.run(schema_size, naturalness=DB_NATURAL)
        file_name = f"{MODEL}{schema_size}{'Natural' if DB_NATURAL else 'Abbreviated'}{'MSchema' if MSCHEMA else ''}{'PreAbstention' if PRE_ABSTENTION else ''}{'PostAbstention' if POST_ABSTENTION else ''}.json"
        save_results(f"{RESULTS_DIR}/{MODEL}/{'Natural' if DB_NATURAL else 'Abbreviated'}/{DATE}/{file_name}", model)
        model.results = {}

def execute_and_analyze_results():
    reporter = Reporter()

    for result_file_name in os.listdir(f"{RESULTS_DIR}/{MODEL}/{'Natural' if DB_NATURAL else 'Abbreviated'}/{DATE}/"):
        path = f"{RESULTS_DIR}/{MODEL}/{'Natural' if DB_NATURAL else 'Abbreviated'}/{DATE}/{result_file_name}"

        if result_file_name == "report.html": continue
        
        with open(path, "r") as file_pointer:
            results = load(file_pointer)

        logger.info(f"Running results of database for {path}.")
        for res in results.values():
            if res['golden_query']:
                if DB_NATURAL == "Normalized":
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
    
    return reporter

if __name__ == "__main__":
    model = get_model()
    run_experiments(model)
    reporter = execute_and_analyze_results()
    reporter.create_report(f"{RESULTS_DIR}/{MODEL}/{'Natural' if DB_NATURAL else 'Abbreviated'}/{DATE}")
