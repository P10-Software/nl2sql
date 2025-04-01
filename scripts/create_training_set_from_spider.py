from src.core.extract_instructions import extract_column_table
from json import load, dump
from os.path import join
from os import walk

PATH_TO_SPIDER_DIR = "../spider_data/spider_data/"


def create_training_set():
    with open(join(PATH_TO_SPIDER_DIR, "train_spider.json"), "r") as train_file:
        spider_train_set = load(train_file)

    ddl_dict = _load_ddl_instructions_for_all_dbs()
    modified_train_set = []
    for question_pair in spider_train_set:
        question = question_pair["question"]
        ddl_instructions = ddl_dict[question_pair["db_id"]]
        used_columns = extract_column_table(question_pair["query"])

def _load_ddl_instructions_for_all_dbs():
    ddl_dict = {}
    database_paths = [join(dirpath,f) for (dirpath, _, filenames) in walk(join(PATH_TO_SPIDER_DIR, "database")) for f in filenames if f.endswith(".sql")]
    for database_path in database_paths:
        db_id = database_path.split("/")[4]
        with open(database_path, "r") as db_file:
            ddl_dict[db_id] = db_file.read()

    return ddl_dict