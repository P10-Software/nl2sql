from src.core.extractive_schema_linking import get_focused_schema, load_schema_linker
import json
import re
from tqdm import tqdm

EVALUATION_DATA_PATH = ".local/SchemaLinker/spider_exsl_dev.json"
SCHEMA_LINKER_PATH = "models/EXSL/OmniSQL_7B_rmc_efficiency_schema_linker_trial_39.pth"
RESULT_DIRECTORY = ".local/experiments/schema_linking_threshold/exsl_omni/column_level/"
THRESHOLD = 0.40

def get_columns_from_schema(schema):
    columns_in_schema = set()
    schema_split = schema.split("# ")
    for table in schema_split[1:]:
        table_name = table.split("\n")[0].split("Table: ")[1]

        # Extract individual column entries inside the brackets
        column_pattern = r"\((.*?)\)"
        columns = re.findall(column_pattern, table)

        for column in columns:
            column_name = column.split(":")[0].strip()
            columns_in_schema.add(f"{table_name} {column_name}")
    return columns_in_schema

if __name__ == "__main__":
    report = []
    precision_sum = 0
    recall_sum = 0
    schema_linker = load_schema_linker(SCHEMA_LINKER_PATH)

    with open(EVALUATION_DATA_PATH, "r") as file:
        dataset = json.load(file)

    for example in tqdm(dataset):
        goal_columns = set(example["goal answer"])
        predicted_schema = get_focused_schema(schema_linker, example["question"], [example["schema"]], example["schema"], THRESHOLD)
        predicted_columns = get_columns_from_schema(predicted_schema)

        correct_predictions = [column for column in predicted_columns if column in goal_columns]

        if len(predicted_columns) == 0:
            precision = 0
        else:
            precision = len(correct_predictions) / len(predicted_columns)
        recall = len(correct_predictions) / len(goal_columns)

        precision_sum += precision
        recall_sum += recall

        report.append({"question": example["question"], "goal columns": list(goal_columns), "predicted columns": list(predicted_columns), "precision": precision, "recall": recall})

    report.append({"Total examples": len(dataset), "Total precision": precision_sum / len(dataset), "Total recall": recall_sum / len(dataset)})

    with open(f"{RESULT_DIRECTORY}trial_39_threshold_{int(THRESHOLD * 100)}_overview.json", "w") as file:
        json.dump(report, file, indent=4)
