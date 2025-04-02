from src.core.extract_instructions import extract_column_table
from json import load, dump
from os.path import join
from os import walk
from sqlalchemy import create_engine
from mschema.schema_engine import SchemaEngine

PATH_TO_SPIDER_DIR = "../spider_data/spider_data/"


def create_training_set():
    with open(join(PATH_TO_SPIDER_DIR, "train_spider.json"), "r") as train_file:
        spider_train_set = load(train_file)
    ddl_dict = _load_ddl_instructions_for_all_dbs()
    modified_train_set = []
    for question_pair in spider_train_set:
        question = question_pair["question"]
        ddl_instructions = ddl_dict[question_pair["db_id"]]
        schema_repeated = ""
        column_table = extract_column_table(question_pair["query"])
        print(column_table)
        goal_columns = []
        for table in column_table.keys():
            for column in column_table[table]:
                goal_columns.append(f"{table}.{column}")

        modified_train_set.append({"input": f"{ddl_instructions}\n To answer: {question}\n We need columns:\n{schema_repeated}", "goal answer": goal_columns})

    return modified_train_set

def _load_ddl_instructions_for_all_dbs():
    ddl_dict = {}
    database_paths = [join(dirpath,f) for (dirpath, _, filenames) in walk(join(PATH_TO_SPIDER_DIR, "database")) for f in filenames if f.endswith(".sqlite")]
    for database_path in database_paths:
        db_id = database_path.split("/")[4]
        db_engine = create_engine(f'sqlite:///{database_path}')
        ddl_dict[db_id] = SchemaEngine(engine=db_engine, db_name=db_id).mschema.to_mschema()

    return ddl_dict