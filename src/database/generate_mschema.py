import os
from sqlalchemy import create_engine
from dotenv import load_dotenv
from mschema.schema_engine import SchemaEngine

load_dotenv()

DB_NAME = os.getenv('DB_NAME')
DB_PATH_ABBREVIATED = os.getenv('DB_PATH_ABBREVIATED')
DB_PATH_NATURAL = os.getenv('DB_PATH_NATURAL')
DB_NATURAL = int(os.getenv('DB_NATURAL', 0))


def generate_mschema():
    """
    Generate schema to M-schema DDL format with additional information
    """
    if DB_NATURAL:
        db_engine = create_engine(f'sqlite:///{DB_PATH_NATURAL}')
    else:
        db_engine = create_engine(f'sqlite:///{DB_PATH_ABBREVIATED}')

    return SchemaEngine(engine=db_engine, db_name=DB_NAME).mschema.to_mschema()


def save_mschema_file():
    """
    Saves the database mschema as a text file in the .local directory
    """
    with open(".local/mschema.txt", "w") as file:
        file.write(generate_mschema())


if __name__ == "__main__":
    save_mschema_file()
