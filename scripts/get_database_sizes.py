import re
import json

DATASET_PATH  = ".local/spider_exsl_dev.json"
SAVE_PATH = ".local/spider_dev_database_info.json"

def extract_db_id(m_schema: str) -> str:
    match = re.search(r'【DB_ID】\s*(\w+)', m_schema)
    return match.group(1) if match else None

def parse_mschema(mschema: str) -> dict:
    table_pattern = r"# Table: (\w+)\s*\[\s*(\(.*?\))\s*\]"
    column_pattern = r"\((.*?)\)"

    table_column_counts = {}
    total_columns = 0

    matches = re.findall(table_pattern, mschema, re.DOTALL)
    
    for table_name, column_block in matches:
        columns = re.findall(column_pattern, column_block)
        num_columns = len(columns)
        table_column_counts[table_name] = num_columns
        total_columns += num_columns

    return {
        "per_table": table_column_counts,
        "total": total_columns
    }

if __name__ == "__main__":
    database_info = {}
    
    with open(DATASET_PATH, "r") as file:
        dataset = json.load(file)

    for example in dataset:
        db_id = extract_db_id(example["schema"])

        if db_id in database_info.keys():
            continue

        database_info[db_id] = parse_mschema(example["schema"])

    with open(SAVE_PATH, "w") as file:
        json.dump(database_info, file, indent=4)
