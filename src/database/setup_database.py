import os
import pyreadstat
import psycopg2
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from tqdm import tqdm
from src.common.logger import get_logger
from src.database.database import get_conn

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

    logger.info("Inserting data from all_trial_metadata and GCMD.")
    for _, files in sas_files.items():
        for sas_file in tqdm(files):
            _read_sas(sas_file)


def init_normalised_db(data_directory: str, table_name_file: str) -> None:
    """
    Inits the database with non abbreviated table and column names, based on the raw SAS DB, 
    recursively adds all sas7bdat files as tables.
    Provide the folder path containing all of the SAS files;

    Parameters:
    - data_directory (str): Root directory to search for SAS files.
    - table_name_file: CSV containing abbreviated and non abbreviated table names
    """

    _create_db()
    _drop_column_label_table()

    sas_files = _find_sas_files(data_directory)

    table_names = pd.read_csv(table_name_file, header=None, names=["old_name", "new_name"])

    logger.info("Inserting data from all_trial_metadata and GCMD with natural language.")
    for _, files in sas_files.items():
        for sas_file in tqdm(files):
            _read_sas_normalised(sas_file, table_names)


def _find_sas_files(directory_path: str):
    sas_files = {}

    for root, _, files in os.walk(directory_path):
        folder_name = os.path.basename(root)
        sas_paths = [os.path.join(root, f) for f in files if f.endswith('.sas7bdat')]

        if sas_paths:
            sas_files[folder_name] = sas_paths

    return sas_files


def _read_sas_normalised(path_to_sas, new_table_names) -> None:
    table_name = os.path.splitext(os.path.basename(path_to_sas))[0]

    # Substitute abbreviated table name with normaised table name
    table_name = new_table_names.loc[new_table_names['old_name'] == table_name, 'new_name'].values[0]

    # meta is all the 'non' visible data, so in our case labels.
    df, meta = pyreadstat.read_sas7bdat(path_to_sas)

    column_names_df = pd.DataFrame({
        'old_names': df.columns.tolist(),
        'new_names': meta.column_labels
    })

    column_mappings = dict(zip(column_names_df['old_names'], column_names_df['new_names']))

    df.rename(columns=column_mappings, inplace=True)
    df.columns = df.columns.map(_column_name_format)

    # Handle duplicate column labels in sponsor_def_value table.
    if table_name == "sponsor_defined_value_in_list":
        df.columns.values[1] = "sponsor_def_submission_value"

    engine = create_engine(f'postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB_NAME}')

    try:
        df.to_sql(table_name, engine, if_exists='replace', index=False)
    except Exception as e:
        logger.error("Error: %s", e)


def _column_name_format(column_name: str) -> str:
    return column_name.replace(' ', '_').lower()


def _read_sas(path_to_sas: str):
    table_name = os.path.splitext(os.path.basename(path_to_sas))[0]

    # meta is all the 'non' visible data, so in our case labels.
    df, meta = pyreadstat.read_sas7bdat(path_to_sas)

    column_names = df.columns

    label_df = pd.DataFrame({
        'table_name': table_name,
        'column_name': column_names,
        'column_label': meta.column_labels
    })

    engine = create_engine(f'postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB_NAME}')

    try:
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        label_df.to_sql('column_label_lookup', engine, if_exists='append', index=False)
        logger.info('Table %s inserted successfully.', table_name)
    except Exception as e:
        logger.error("Error: %s", e)


def _create_db():
    try:
        conn = psycopg2.connect(dbname="postgres", user=USER, password=PASSWORD, host=HOST, port=PORT)
        conn.autocommit = True  # Enable auto-commit to execute CREATE DATABASE outside of a transaction
        cur = conn.cursor()

        cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{DB_NAME}'")
        exists = cur.fetchone()

        if exists:
            logger.info("Database %s already exists.", DB_NAME)

            # Terminate active connections before dropping the database
            cur.execute(f"""
                SELECT pg_terminate_backend(pg_stat_activity.pid)
                FROM pg_stat_activity
                WHERE pg_stat_activity.datname = '{DB_NAME}'
                AND pid <> pg_backend_pid();
            """)

            cur.execute("DROP DATABASE {DB_NAME}")
            logger.info("Database %s dropped successfully.", DB_NAME)

        cur.execute(f"CREATE DATABASE {DB_NAME}")
        logger.info("Database %s created successfully.", DB_NAME)

        cur.close()
        conn.close()
    except Exception as e:
        logger.error("Error creating database: %s", e)


def _drop_column_label_table():
    """
    Drop the tables column_label_lookup table to avoid duplicates 
    """
    try:
        conn = get_conn()
        conn.autocommit = True
        cur = conn.cursor()

        cur.execute("DROP TABLE IF EXISTS column_label_lookup")

        cur.close()
        conn.close()
    except Exception as e:
        logger.error("Error dropping the table column_label_lookup. %s", e)


if __name__ == '__main__':
    try:
        init_db("NOVO_SAS_DATA")
        init_normalised_db("NOVO_SAS_DATA", "../data/table_names_normalised.csv") # Normalised DB
    except Exception as e:
        logger.error("Failed: %s", e)
