from os.path import join, isfile
from os import listdir, getenv
import psycopg2
from libs.SNAILS.snails_naturalness_classifier import CanineIdentifierClassifier
from dotenv import load_dotenv
from json import load, dump

# Database credentials
load_dotenv()

USER = getenv('PG_USER')
PASSWORD = getenv('PG_PASSWORD')
HOST = getenv('PG_HOST')
PORT = getenv('PG_PORT')
DB_NAME = getenv('DB_NAME')

SPIDER_DIR_PATH = "Src/Spider2Lite/resource/databases/sqlite"

def evaluate_novo_database_naturalness():
    conn = psycopg2.connect(dbname=DB_NAME, user=USER, password=PASSWORD, host=HOST, port=PORT)
    conn.autocommit = False  # Enable auto-commit to execute CREATE DATABASE outside of a transaction
    cur = conn.cursor()

    #get table names
    cur.execute('select "Table_Name" from column_label_lookup;')
    table_identifiers = {result[0] for result in cur.fetchall()}

    #get normalised table names
    normalized_table_identifiers = set()
    with open("normalized_table_names.txt", "r", encoding="utf-8") as file:
        for line in file:
            normalized_table_identifiers.add(line.strip())

    #get column names
    cur.execute('select "Column_Name" from column_label_lookup;')
    column_identifiers = {result[0] for result in cur.fetchall()}

    #get column labels
    cur.execute('select "Column_Label" from column_label_lookup;')
    column_labels = {result[0] for result in cur.fetchall()}


    naturalness_table_results = evaluate_naturalness(table_identifiers)
    naturalness_normalized_table_results = evaluate_naturalness(normalized_table_identifiers)
    naturalness_column_results = evaluate_naturalness(column_identifiers)
    naturalness_column_label_results = evaluate_naturalness(column_labels)

    with open("table_naturalness_results.json", "w") as file:
        dump(naturalness_table_results, file, indent=4)

    with open("normalized_table_naturalness_results.json", "w") as file:
        dump(naturalness_normalized_table_results, file, indent=4)

    with open("column_naturalness_results.json", "w") as file:
        dump(naturalness_column_results, file, indent=4)

    with open("column_label_naturalness_results.json", "w") as file:
        dump(naturalness_column_label_results, file, indent=4)

    cur.close()

def evaluate_spider_2_naturalness():
    db_paths = [join(SPIDER_DIR_PATH, db_path) for db_path in listdir(SPIDER_DIR_PATH)]
    column_identifiers = set()
    table_identifiers = set()


    for db_path in db_paths:
        table_paths = [join(db_path, file) for file in listdir(db_path) if isfile(join(db_path, file)) and file.endswith(".json")]

        # Get all columns (including duplicates)
        for table_path in table_paths:
            with open(table_path, "r") as file:
                table_dict = load(file)
            column_identifiers.update(table_dict["column_names"])
            table_identifiers.add(table_dict["table_name"])

    table_naturalness_results = evaluate_naturalness(table_identifiers)
    column_naturalness_results = evaluate_naturalness(column_identifiers)

    with open("spider2_table_naturalness_results.json", "w") as file:
        dump(table_naturalness_results, file, indent=4)
    
    with open("spider2_column_naturalness_results.json", "w") as file:
        dump(column_naturalness_results, file, indent=4)

def evaluate_kaggle_dbqa_naturalness():
    with open("KaggleDBQA_tables.json", "r") as file:
        dataset = load(file)

    tables = set()
    columns = set()
    for database in dataset:
        tables.update(database["table_names"])
        columns.update({column[1] for column in database["column_names"] if column[1] != "*"})

    table_naturalness_results = evaluate_naturalness(tables)
    column_naturalness_results = evaluate_naturalness(columns)
    
    with open("kaggledbqa_table_naturalness_results.json", "w") as file:
        dump(table_naturalness_results, file, indent=4)
    
    with open("kaggledbqa_column_naturalness_results.json", "w") as file:
        dump(column_naturalness_results, file, indent=4)

def evaluate_naturalness(indentifiers):
    classifier = CanineIdentifierClassifier()

    classifications = dict()
    for identifier in indentifiers:
        classifications[identifier] = classifier.classify_identifier(identifier)[0]["label"]

    distribution_of_classifications = {"distribution": {
        "N1": len([value for value in classifications.values() if value == "N1"]),
        "N2": len([value for value in classifications.values() if value == "N2"]),
        "N3": len([value for value in classifications.values() if value == "N3"])
    }}
    
    return distribution_of_classifications | classifications

def main():
    evaluate_novo_database_naturalness()
    evaluate_spider_2_naturalness()
    evaluate_kaggle_dbqa_naturalness()

if __name__ == "__main__":
    main()
