import json
import re

RESULT_FILE = ".local/experiment3_column/trial_39_threshold_40_overview.json"
TASK_FILE = ".local/spider_exsl_dev.json"
DB_INFO_FILE = ".local/spider_dev_database_info.json"
OUTPUT_FILE = ".local/experiment3_column/trial_39_threshold_40_overview_with_reduction.json"

def extract_db_id(m_schema: str) -> str:
    match = re.search(r'【DB_ID】\s*(\w+)', m_schema)
    return match.group(1) if match else None

with open(RESULT_FILE, "r") as file:
    results = json.load(file)

with open(TASK_FILE, "r") as file:
    tasks = json.load(file)

with open(DB_INFO_FILE, "r") as file:
    db_info = json.load(file)

reduction_sum_column = 0
reduction_sum_table = 0
for result, task in zip(results[:-1], tasks):
    db_id = extract_db_id(task["schema"])

    filtered_schema_size_column = len(result["predicted columns"])
    relative_schema_reduction_column = (db_info[db_id]["total"] - filtered_schema_size_column) / db_info[db_id]["total"]
    filtered_schema_size_table = 0
    for table in {column.split(" ")[0] for column in result["predicted columns"]}:
        filtered_schema_size_table += db_info[db_id]["per_table"][table]

    relative_schema_reduction_table = (db_info[db_id]["total"] - filtered_schema_size_table) / db_info[db_id]["total"]

    result["relative reduction column"] = relative_schema_reduction_column
    result["relative reduction table"] = relative_schema_reduction_table

    reduction_sum_column += relative_schema_reduction_column
    reduction_sum_table += relative_schema_reduction_table

examples_in_list = len(results) - 1
results[-1]["Average reduction column"] = reduction_sum_column / examples_in_list
results[-1]["Average reduction table"] = reduction_sum_table / examples_in_list

with open(OUTPUT_FILE, "w") as file:
    json.dump(results, file, indent=4)