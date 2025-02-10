import os
import psycopg2
from libs.SNAILS.snails_naturalness_classifier import CanineIdentifierClassifier
from dotenv import load_dotenv
import json

# Database credentials
load_dotenv()

USER = os.getenv('PG_USER')
PASSWORD = os.getenv('PG_PASSWORD')
HOST = os.getenv('PG_HOST')
PORT = os.getenv('PG_PORT')
DB_NAME = os.getenv('DB_NAME')

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
    conn = psycopg2.connect(dbname=DB_NAME, user=USER, password=PASSWORD, host=HOST, port=PORT)
    conn.autocommit = True  # Enable auto-commit to execute CREATE DATABASE outside of a transaction
    cur = conn.cursor()

    #get table names
    cur.execute('select "Table_Name" from column_label_lookup;')
    table_identifiers = {result[0] for result in cur.fetchall()}

    #get column names
    cur.execute('select "Column_Name" from column_label_lookup;')
    column_identifiers = {result[0] for result in cur.fetchall()}

    #get column labels
    cur.execute('select "Column_Label" from column_label_lookup;')
    column_labels = {result[0] for result in cur.fetchall()}


    naturalness_table_results = evaluate_naturalness(table_identifiers)
    naturalness_column_results = evaluate_naturalness(column_identifiers)
    naturalness_column_label_results = evaluate_naturalness(column_labels)

    with open("table_naturalness_results.json", "w") as file:
        json.dump(naturalness_table_results, file, indent=4)

    with open("column_naturalness_results.json", "w") as file:
        json.dump(naturalness_column_results, file, indent=4)

    with open("column_label_naturalness_results.json", "w") as file:
        json.dump(naturalness_column_label_results, file, indent=4)

if __name__ == "__main__":
    main()
