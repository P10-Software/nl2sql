from src.core.extractive_schema_linking import load_schema_linker, get_focused_schema
from src.core.schema_chunking import chunk_mschema
from experiments.schema_linking_threshold import get_columns_from_schema
from tqdm import tqdm
import json

EVALUATION_DATA_PATH = ".local/SchemaLinker/spider_exsl_all_to_single_test.json"
SCHEMA_LINKER_PATH = "models/EXSL/OmniSQL_7B_rmc_efficiency_schema_linker_trial_39.pth"
RESULT_DIRECTORY = ".local/experiments/schema_linking/spider/exsl_omni/column_level/"
TABLES_PER_CHUNK = 1
WITH_RELATIONS = False

def get_table_names_from_schema(schema):
    schema_split = schema.split("# ")
    schema_tables = [table.split("\n")[0].split("Table: ")[1] for table in schema_split[1:]]
    return schema_tables

def evaluate_extractive_schema_linking(schema_linker_path: str, dataset: list, k: int = 0):
    schema_linker = load_schema_linker(schema_linker_path)
    sum_recall = 0
    sum_precision = 0
    report = []

    for example in tqdm(dataset):
        #chunks = chunk_mschema(example["schema"], schema_linker.tokenizer, False, k)
        chunks = [example["schema"]]

        goal_columns = set(example["goal answer"])
        #goal_tables = {column.split(" ")[0] for column in goal_columns}

        # Predict focused schema
        predicted_schema = get_focused_schema(schema_linker, example["question"], chunks, example["schema"], 0.15)
        #predicted_tables = get_table_names_from_schema(predicted_schema)
        predicted_columns = get_columns_from_schema(predicted_schema)

        correct_predictions = [table for table in predicted_columns if table in goal_columns]

        if len(predicted_columns) == 0:
            precision = 0
        else:
            precision = len(correct_predictions) / len(predicted_columns)
        recall = len(correct_predictions) / len(goal_columns)
    
        sum_precision += precision
        sum_recall += recall

        report.append({"question": example["question"], "goal columns": list(goal_columns), "predicted tables": list(predicted_columns), "recall": recall, "precision": precision})

    report.append({"Dataset Size": len(dataset), "Total recall": sum_recall / len(dataset), "Total precision": sum_precision / len(dataset)})
    return report

if __name__ == "__main__":
    with open(EVALUATION_DATA_PATH, "r") as eval_file:
        eval_set = json.load(eval_file)
    
    report = evaluate_extractive_schema_linking(SCHEMA_LINKER_PATH, eval_set, TABLES_PER_CHUNK)

    with open(f"{RESULT_DIRECTORY}no_chunking_{TABLES_PER_CHUNK}_overview.json", "w") as file:
        json.dump(report, file, indent=4)