import os
import sqlite3
import pyreadstat
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from src.common.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

DB_PATH_ABBREVIATED = os.getenv('DB_PATH_ABBREVIATED')
DB_PATH_NATURAL = os.getenv('DB_PATH_NATURAL')


def init_db(data_directory: str):
    """
    Inits the database, based on the raw SAS DB, recursively adds all sas7bdat files as tables.
    Provide the folder path containing all of the SAS files;

    Parameters:
    - data_directory (str): Root directory to search for SAS files.
    """

    _delete_db(f'{DB_PATH_ABBREVIATED}')

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

    _delete_db(f'{DB_PATH_NATURAL}')

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
    df.columns = df.columns.map(column_name_format)

    # Handle duplicate column labels in sponsor_def_value table.
    if table_name == "sponsor_defined_value_in_list":
        df.columns.values[2] = "sponsor_defined_submission_value"

    conn = sqlite3.connect(f'{DB_PATH_NATURAL}')

    try:
        df.to_sql(table_name, conn, if_exists='replace', index=False)
    except Exception as e:
        logger.error("Error: %s", e)

    conn.close()


def column_name_format(column_name: str) -> str:
    """ Ensure that the columns comply with database naming rules."""
    return column_name.strip().replace(' ', '_').replace('-', '_').replace('.', '').lower()


def _read_sas(path_to_sas: str):
    table_name = os.path.splitext(os.path.basename(path_to_sas))[0]

    # meta is all the 'non' visible data, so in our case labels.
    df, meta = pyreadstat.read_sas7bdat(path_to_sas)

    df.columns = df.columns.map(column_name_format)

    column_names = df.columns

    label_df = pd.DataFrame({
        'table_name': table_name,
        'column_name': column_names,
        'column_label': meta.column_labels
    })

    conn = sqlite3.connect(f'{DB_PATH_ABBREVIATED}')

    try:
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        label_df.to_sql('column_label_lookup', conn, if_exists='append', index=False)
    except Exception as e:
        logger.error("Error: %s", e)

    conn.close()


def _delete_db(db_path: str):
    try:
        if os.path.exists(f"{db_path}"):
            os.remove(f"{db_path}")
            logger.info("Deleted database from path: %s", db_path)
    except Exception as e:
        logger.error("Error deleting database file %s: %s", db_path, e)


if __name__ == '__main__':
    try:
        init_db(".local/NOVO_SAS_DATA")
        init_normalised_db(".local/NOVO_SAS_DATA", ".local/table_names_normalised.csv")
    except Exception as e:
        logger.error("Failed to setup SQLite databases: %s", e)
