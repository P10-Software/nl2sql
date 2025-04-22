from src.core.extract_instructions import extract_column_table
from json import load, dump
from os.path import join, basename
from os import walk
from sqlalchemy import create_engine
from mschema.schema_engine import SchemaEngine
import re

PATH_TO_SPIDER_DIR = "../spider_data/spider_data/"


def create_training_set():
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
            column_table = extract_column_table(question_pair["query"], schema_dict[question_pair["db_id"]]["db_path"])
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
    # Column name to lower
    updated_schema = re.sub(r"\(\s*(\w+):(\w+)", lambda m: "(" + m.group(1).lower() + ":" + m.group(2), schema)
    # Table name to lower
    updated_schema = re.sub(r"(# Table:\s*)([A-Z])(\w+)", lambda m: m.group(1) + m.group(2).lower() + m.group(3), updated_schema)
    return updated_schema

if __name__ == "__main__":
    valid_training_set, invalid_training_set = create_training_set()
    with open(".local/spider_exsl_train.json", "w") as file:
        dump(valid_training_set, file, indent=4)

    with open(".local/erroneous_spider_exsl_train.json", "w") as file:
        dump(invalid_training_set, file, indent=4)