from src.core.extractive_schema_linking import load_schema_linker, get_focused_schema
from src.core.schema_chunking import mschema_to_k_chunks, chunk_mschema
from tqdm import tqdm
import json

EVALUATION_DATA_PATH = ".local/SchemaLinker/metadata_exsl.json"
SCHEMA_LINKER_PATH = "models/EXSL/OmniSQL_7B_optimal_params_coarse_grained_schema_linker_spider.pth"
RESULT_DIRECTORY = ".local/experiments/schema_linking/metadata/exsl_omni/"
NUMBER_OF_CHUNKS = 1

def get_column_names_from_schema(schema):
    schema_split = schema.split("# ")
    schema_tables = [table.split("\n")[0].split("Table: ")[1] for table in schema_split[1:]]
    return schema_tables

def evaluate_extractive_schema_linking(schema_linker_path: str, dataset: list, chunk_amount: int = 0):
    schema_linker = load_schema_linker(schema_linker_path)
    sum_recall = 0
    sum_precision = 0
    report = []

    for example in tqdm(dataset):
        #chunks = mschema_to_k_chunks(example["schema"], schema_linker.tokenizer, chunk_amount)
        #chunks = chunk_mschema(example["schema"], schema_linker, False)
        chunks = [example["schema"]]

        goal_columns = example["goal answer"]
        goal_tables = {column.split(" ")[0] for column in goal_columns}

        # Predict focused schema
        predicted_schema = get_focused_schema(schema_linker, example["question"], chunks, example["schema"])
        predicted_tables = get_column_names_from_schema(predicted_schema)

        correct_predictions = [table for table in predicted_tables if table in goal_tables]

        if len(predicted_tables) == 0:
            precision = 0
        else:
            precision = len(correct_predictions) / len(predicted_tables)
        recall = len(correct_predictions) / len(goal_tables)
    
        sum_precision += precision
        sum_recall += recall

        report.append({"question": example["question"], "goal tables": list(goal_tables), "predicted tables": predicted_tables, "recall": recall, "precision": precision})

    report.append({"Dataset Size": len(dataset), "Total recall": sum_recall / len(dataset), "Total precision": sum_precision / len(dataset)})
    return report

if __name__ == "__main__":
    with open(EVALUATION_DATA_PATH, "r") as eval_file:
        eval_set = json.load(eval_file)
    
    report = evaluate_extractive_schema_linking(SCHEMA_LINKER_PATH, eval_set, NUMBER_OF_CHUNKS)

    with open(f"{RESULT_DIRECTORY}chunks_{NUMBER_OF_CHUNKS}_overview.json", "w") as file:
        json.dump(report, file, indent=4)