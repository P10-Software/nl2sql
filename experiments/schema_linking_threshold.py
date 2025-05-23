from src.core.extractive_schema_linking import get_focused_schema, load_schema_linker
import json
from tqdm import tqdm

EVALUATION_DATA_PATH = ".local/SchemaLinker/spider_exsl_dev.json"
SCHEMA_LINKER_PATH = "models/EXSL/OmniSQL_7B_rmc_efficiency_schema_linker_trial_33.pth"
RESULT_DIRECTORY = ".local/experiments/schema_linking_threshold/exsl_omni/"
THRESHOLD = 0.40

def get_table_names_from_schema(schema):
    schema_split = schema.split("# ")
    schema_tables = [table.split("\n")[0].split("Table: ")[1] for table in schema_split[1:]]
    return schema_tables

if __name__ == "__main__":
    report = []
    precision_sum = 0
    recall_sum = 0
    schema_linker = load_schema_linker(SCHEMA_LINKER_PATH)

    with open(EVALUATION_DATA_PATH, "r") as file:
        dataset = json.load(file)

    for example in tqdm(dataset):
        goal_tables = {column.split(" ")[0] for column in example["goal answer"]}
        predicted_schema = get_focused_schema(schema_linker, example["question"], [example["schema"]], example["schema"], THRESHOLD)
        predicted_tables = get_table_names_from_schema(predicted_schema)

        correct_predictions = [table for table in predicted_tables if table in goal_tables]

        if len(predicted_tables) == 0:
            precision = 0
        else:
            precision = len(correct_predictions) / len(predicted_tables)
        recall = len(correct_predictions) / len(goal_tables)

        precision_sum += precision
        recall_sum += recall

        report.append({"question": example["question"], "goal tables": list(goal_tables), "predicted tables": predicted_tables, "precision": precision, "recall": recall})

    report.append({"Total examples": len(dataset), "Total precision": precision_sum / len(dataset), "Total recall": recall_sum / len(dataset)})

    with open(f"{RESULT_DIRECTORY}trial_33_threshold_{int(THRESHOLD * 100)}_overview.json", "w") as file:
        json.dump(report, file, indent=4)
