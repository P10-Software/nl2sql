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
    Generate M-schema, containing additional information compared to DDL
    """

    db_engine = create_engine(f'sqlite:///{DB_PATH_NATURAL}')
    with open(".local/mschema_natural.txt", "w") as file:
        file.write(SchemaEngine(engine=db_engine, db_name=DB_NAME).mschema.to_mschema())

    print("Successfully generated M-Schema for natural database")

    db_engine = create_engine(f'sqlite:///{DB_PATH_ABBREVIATED}')
    with open(".local/mschema_abbreviated.txt", "w") as file:
        file.write(SchemaEngine(engine=db_engine, db_name=DB_NAME).mschema.to_mschema())

    print("Successfully generated M-Schema for abbreviated database")


if __name__ == "__main__":
    generate_mschema()
