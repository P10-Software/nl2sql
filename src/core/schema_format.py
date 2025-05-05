import os
from dotenv import load_dotenv
from src.database.database import execute_query

load_dotenv()

DB_NAME = os.getenv('DB_NAME')
DB_NATURAL = bool(int(os.getenv('DB_NATURAL', 0)))


def schema_filtering(relevant_tables: set) -> str:
    """
    Receives a set of relevant table names from the schema linking process.
    Creates an M-Schema containing only relevant tables.
    
    Args:
        - relevant_tables (set)
    """

    mschema = get_mschema()

    return ""


def get_mschema():
    """
    Read database m-schema from file.
    """
    with open(f".local/mschema_{DB_NAME}_{'natural' if DB_NATURAL else 'abbreviated'}.txt", "r") as file:
        return file.read()


def get_DDL() -> str:
    """
    Get database DDL instructions from the database.
    """
    query = """
        SELECT sql
        FROM sqlite_master
        WHERE type IN ('table', 'index', 'view', 'trigger')
    """
    result = execute_query(query)

    ddl_statements = [row[0] for row in result]
    full_ddl = ";\n\n".join(ddl_statements) + ";"

    return full_ddl
