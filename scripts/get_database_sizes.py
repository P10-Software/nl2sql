import re
import json

DATASET_PATH = ".local/spider_exsl_all_to_single_test.json"
SAVE_PATH = ".local/spider_test_database_info.json"

def extract_db_id(m_schema: str) -> str:
    match = re.search(r'【DB_ID】\s*(\w+)', m_schema)
    return match.group(1) if match else None

def extract_table_blocks(mschema: str):
    """Extract each table name and its column block (everything between square brackets)."""
    table_blocks = []
    pattern = r"# Table: (\w+)\s*\[\s*(.*?)\s*\](?=\s*# Table:|\s*【|\Z)"  # match until next table or end
    matches = re.findall(pattern, mschema, re.DOTALL)
    return matches

def split_columns(block: str):
    """Split column definitions on top-level commas only."""
    columns = []
    current = ''
    bracket_depth = 0
    paren_depth = 0

    for char in block:
        if char == '[':
            bracket_depth += 1
        elif char == ']':
            bracket_depth -= 1
        elif char == '(':
            paren_depth += 1
        elif char == ')':
            paren_depth -= 1

        if char == ',' and bracket_depth == 0 and paren_depth == 0:
            if current.strip():
                columns.append(current.strip())
            current = ''
        else:
            current += char

    if current.strip():
        columns.append(current.strip())
    return columns

def parse_mschema(mschema: str) -> dict:
    table_column_counts = {}
    total_columns = 0

    table_blocks = extract_table_blocks(mschema)

    for table_name, block in table_blocks:
        # Extract all top-level ( ... ) blocks inside the square brackets
        column_defs = re.findall(r"\([^\(\)]*?\)", block, re.DOTALL)
        num_columns = len(column_defs)
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
        if not db_id or db_id in database_info:
            continue
        database_info[db_id] = parse_mschema(example["schema"])

    with open(SAVE_PATH, "w") as file:
        json.dump(database_info, file, indent=4)
