from src.core.extract_instructions import extract_column_table
from json import load, dump
from os.path import join, basename
from os import walk
from sqlalchemy import create_engine
from mschema.schema_engine import SchemaEngine
import re
import os

PATH_TO_SPIDER_DIR = "../spider_data/spider_data/"


def create_training_set():
    with open(join(PATH_TO_SPIDER_DIR, "train_spider.json"), "r") as train_file:
        spider_train_set = load(train_file)

    schema_dict = _load_schema_for_all_dbs()
    modified_train_set = []

    for question_pair in spider_train_set:
        question = question_pair["question"]
        schema = schema_dict[question_pair["db_id"]]["schema"]
        schema_repeated = _extract_columns_in_schema(schema)
        
        # Extract goal columns from query
        column_table = extract_column_table(question_pair["query"], schema_dict[question_pair["db_id"]]["db_path"])
        goal_columns = []
        for table in column_table.keys():
            for column in column_table[table]:
                goal_columns.append(f"{table}.{column}")

        modified_train_set.append({"input": f"{schema}\nTo answer: {question}\nWe need columns:\n{schema_repeated}", "goal answer": goal_columns})

    return modified_train_set

def _load_schema_for_all_dbs():
    ddl_dict = {}
    database_paths = [join(dirpath,f) for (dirpath, _, filenames) in walk(join(PATH_TO_SPIDER_DIR, "database")) for f in filenames if f.endswith(".sqlite")]
    for database_path in database_paths:
        db_id = basename(database_path).split(".")[0]
        db_engine = create_engine(f'sqlite:///{database_path}')
        ddl_dict[db_id] = {"schema": SchemaEngine(engine=db_engine, db_name=db_id).mschema.to_mschema(), "db_path": database_path}
    return ddl_dict

def _extract_columns_in_schema(mschema: str) -> str:
    columns_in_schema = ""
    
    # Split schema by tables
    table_sections = re.split(r"# Table: (\w+)", mschema)[1:]  
    for i in range(0, len(table_sections), 2):
        table_name = table_sections[i].strip()  
        columns_section = table_sections[i + 1]

        # Extract column names
        column_matches = re.findall(r"\(\s*(\w+):", columns_section)
        for column in column_matches:
            columns_in_schema += f"<< {table_name} {column} >>\n"

    return columns_in_schema

if __name__ == "__main__":
    training_set = create_training_set()
    with open(".local/spider_exsl_train.json", "w") as file:
        dump(training_set, file, indent=4)