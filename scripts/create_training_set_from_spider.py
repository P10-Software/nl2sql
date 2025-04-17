import re
from json import load, dump
from os.path import join, basename
from os import walk
from sqlalchemy import create_engine
from mschema.schema_engine import SchemaEngine
from scripts.extract_instructions import extract_column_table

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
                    goal_columns.append(f"{table} {column}")
            goal_columns = [s[0].lower() + s[1:] if s else s for s in goal_columns]

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
        schema_dict[db_id] = {"schema": _lowercase_column_names(SchemaEngine(engine=db_engine, db_name=db_id).mschema.to_mschema()), "db_path": database_path}
    return schema_dict

def _lowercase_column_names(schema: str) -> str:
    def lowercase_column(match):
        column_name = match.group(1)
        if column_name[0].isupper():
            column_name = column_name[0].lower() + column_name[1:]
        return f"({column_name}:{match.group(2)}"

    # Apply to every column inside (column_name:TYPE, ...)
    updated_schema = re.sub(r"\(\s*(\w+):(\w+)", lowercase_column, schema)
    return updated_schema

if __name__ == "__main__":
    valid_training_set, invalid_training_set = create_training_set()
    with open(".local/spider_exsl_train.json", "w") as file:
        dump(valid_training_set, file, indent=4)

    with open(".local/erroneous_spider_exsl_train.json", "w") as file:
        dump(invalid_training_set, file, indent=4)
