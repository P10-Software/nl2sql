import os
import pyreadstat
import psycopg2
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from src.common.logger import get_logger

# Database credentials
load_dotenv()
logger = get_logger(__name__)

USER = os.getenv('PG_USER')
PASSWORD = os.getenv('PG_PASSWORD')
HOST = os.getenv('PG_HOST')
PORT = os.getenv('PG_PORT')
DB_NAME = os.getenv('DB_NAME')


def init_db(data_directory: str):
    """
    Inits the database, based on the raw SAS DB, recursively adds all sas7bdat files as tables.
    Provide the folder path containing all of the SAS files;

    Parameters:
    - data_directory (str): Root directory to search for SAS files.
    """

    _create_db()
    _drop_column_label_table()

    sas_files = _find_sas_files(data_directory)

    for _, files in sas_files.items():
        for sas_file in files:
            _read_sas(sas_file)


def _find_sas_files(directory_path: str):
    sas_files = {}

    for root, _, files in os.walk(directory_path):
        folder_name = os.path.basename(root)
        sas_paths = [os.path.join(root, f) for f in files if f.endswith('.sas7bdat')]

        if sas_paths:
            sas_files[folder_name] = sas_paths

    return sas_files


def _read_sas(path_to_sas: str):
    table_name = os.path.splitext(os.path.basename(path_to_sas))[0]

    # meta is all the 'non' visible data, so in our case labels.
    df, meta = pyreadstat.read_sas7bdat(path_to_sas)

    column_names = df.columns

    label_df = pd.DataFrame({
        'Table_Name': table_name,
        'Column_Name': column_names,
        'Column_Label': meta.column_labels
    })

    engine = create_engine(f'postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB_NAME}')

    try:
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        label_df.to_sql('column_label_lookup', engine, if_exists='append', index=False)
        logger.info(f'Table {table_name} inserted successfully.')
    except Exception as e:
        logger.error("Error: ", e)


def _create_db():
    try:
        conn = psycopg2.connect(dbname="postgres", user=USER, password=PASSWORD, host=HOST, port=PORT)
        conn.autocommit = True  # Enable auto-commit to execute CREATE DATABASE outside of a transaction
        cur = conn.cursor()

        cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{DB_NAME}'")
        exists = cur.fetchone()

        if not exists:
            cur.execute(f"CREATE DATABASE {DB_NAME}")
            logger.info(f"Database '{DB_NAME}' created successfully.")
        else:
            logger.info(f"Database '{DB_NAME}' already exists.")

        cur.close()
        conn.close()
    except Exception as e:
        logger.error("Error creating database:", e)


def _drop_column_label_table():
    """
    Drop the tables column_label_lookup table to avoid duplicates 
    """
    try:
        conn = psycopg2.connect(dbname=DB_NAME, user=USER, password=PASSWORD, host=HOST, port=PORT)
        conn.autocommit = True
        cur = conn.cursor()

        cur.execute("DROP TABLE IF EXISTS column_label_lookup")

        cur.close()
        conn.close()
    except Exception as e:
        logger.error("Error dopping the table column_label_lookup. ", e)


if __name__ == '__main__':
    data_directory = input("Provide path to the data folder: ").strip()

    if os.path.isdir(data_directory):
        try:
            init_db(data_directory)
        except Exception as e:
            logger.error("Failed: ", e)
    else:
        logger.error("Provided path is not a directory.")
