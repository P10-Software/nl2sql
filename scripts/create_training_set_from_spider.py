import re
from json import load, dump
from os.path import join, basename
from os import walk
from sqlalchemy import create_engine
from mschema.schema_engine import SchemaEngine
from scripts.extract_instructions import extract_column_table

PATH_TO_SPIDER_DIR = "../spider_data/spider_data/"


def create_training_set(replace_all_with_single: bool = False):
    with open(join(PATH_TO_SPIDER_DIR, "train_spider.json"), "r") as train_file:
        spider_train_set = load(train_file)

    schema_dict = _load_schema_for_all_dbs()
    valid_train_set = []
    invalid_train_set = []

    for question_pair in spider_train_set:
        question = question_pair["question"]
        schema = schema_dict[question_pair["db_id"]]["schema"]
        
        # Extract goal columns from query
        try:
            column_table = extract_column_table(question_pair["query"], schema_dict[question_pair["db_id"]]["db_path"], replace_all_with_single)
            goal_columns = []
            for table in column_table.keys():
                for column in column_table[table]:
                    goal_columns.append(f"{table.lower()} {column.lower()}")

            valid_train_set.append({"question": question, "schema": schema, "goal answer": goal_columns})
        except:
            invalid_train_set.append({"question": question, "query": question_pair["query"]})

    return valid_train_set, invalid_train_set

def _load_schema_for_all_dbs():
    schema_dict = {}
    database_paths = [join(dirpath,f) for (dirpath, _, filenames) in walk(join(PATH_TO_SPIDER_DIR, "database")) for f in filenames if f.endswith(".sqlite")]
    for database_path in database_paths:
        db_id = basename(database_path).split(".")[0]
        db_engine = create_engine(f'sqlite:///{database_path}')
        schema_dict[db_id] = {"schema": _lowercase_column_and_table_names(SchemaEngine(engine=db_engine, db_name=db_id).mschema.to_mschema()), "db_path": database_path}
    return schema_dict

def _lowercase_column_and_table_names(schema: str) -> str:
    # Lowercase table names in "# Table: ..."
    output = re.sub(r"(# Table:\s*)([A-Za-z_][\w]*)", lambda m: m.group(1) + m.group(2).lower(), schema)

    # Lowercase column names (first token in each parentheses)
    output = re.sub(
        r"\(\s*([A-Za-z_][\w]*)",
        lambda m: "(" + m.group(1).lower(),
        output
    )

    # Find the Foreign keys section, keep header as-is, lowercase rest
    def fix_foreign_keys_section(match):
        header = "【Foreign keys】"
        body = match.group(1)
        body_fixed = re.sub(r"\b([A-Za-z_][\w]*)", lambda m: m.group().lower(), body)
        return f"{header}\n{body_fixed}"

    return re.sub(r"【Foreign keys】\n([\s\S]*)", fix_foreign_keys_section, output)

def exsl_dataset_to_dts_format(exsl_dataset):
    dts_dataset = []
    for example in exsl_dataset:
        dts_dataset.append(
            {
                "question": exsl_dataset["question"],
                "database_schema": m_schema_to_ddl(exsl_dataset["schema"]),
                "correct_tables": ", ".join(""),
                "correct_columns": ", ".join(""),
            }
        )

def m_schema_to_ddl(m_schema: str) -> str:
    lines = m_schema.strip().splitlines()
    ddl_statements = []
    tables = {}
    foreign_keys = []

    current_table = None

    for line in lines:
        line = line.strip()
        if line.startswith("# Table:"):
            current_table = line.split(":")[1].strip()
            tables[current_table] = []
        elif line.startswith("["):
            continue  # Start of column block
        elif line.startswith("]"):
            current_table = None
        elif "=" in line and "." in line:
            foreign_keys.append(line)
        elif current_table and line.startswith("("):
            # Extract column info
            parts = line.strip("(),").split(",")
            col_def = {}
            if len(parts) > 1 and parts[1].lower().strip() == "primary key":
                    col_def["primary_key"] = True
            if ":" in parts[0].strip():
                col_name, col_type = map(str.strip, parts[0].strip().split(":"))
                col_def["name"] = col_name
                col_def["type"] = col_type
            tables[current_table].append(col_def)

    # Generate CREATE TABLE statements
    for table, cols in tables.items():
        col_lines = []
        pk_lines = []
        for col in cols:
            col_line = f"{col['name']} {col['type']}"
            if col.get("primary_key"):
                pk_lines.append(col["name"])
            col_lines.append(col_line)

        ddl = f"CREATE TABLE {table} (\n"
        ddl += "    " + ",\n    ".join(col_lines)
        if pk_lines:
            ddl += ",\n    PRIMARY KEY (" + ", ".join(pk_lines) + ")"
        ddl += "\n);"
        ddl_statements.append(ddl)

    # Add foreign key constraints
    for fk in foreign_keys:
        left, right = fk.split("=")
        left_table, left_col = left.strip().split(".")
        right_table, right_col = right.strip().split(".")
        ddl = (f"ALTER TABLE {right_table} "
               f"ADD FOREIGN KEY ({right_col}) "
               f"REFERENCES {left_table}({left_col});")
        ddl_statements.append(ddl)

    return "\n\n".join(ddl_statements)


if __name__ == "__main__":
    valid_training_set, invalid_training_set = create_training_set(False)
    with open(".local/spider_exsl_train.json", "w") as file:
        dump(valid_training_set, file, indent=4)

    with open(".local/erroneous_spider_exsl_train.json", "w") as file:
        dump(invalid_training_set, file, indent=4)
