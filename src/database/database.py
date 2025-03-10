import os
import sqlite3
from dotenv import load_dotenv
from src.common.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

DB_NAME = os.getenv('DB_NAME')
DB_PATH_ABBREVIATED = os.getenv('DB_PATH_ABBREVIATED')
DB_PATH_NATURAL = os.getenv('DB_PATH_NATURAL')
DB_NATURAL = int(os.getenv('DB_NATURAL', 0))


def execute_query(query: str):
    """
    Executes a query on the database and returns the raw answer

    Args:
    - query: SQL query as a string.

    Returns:
    - The result of executing the SQL.
    """

    result = []
    conn = get_conn()
    cur = conn.cursor()

    if conn:
        try:
            cur.execute(query)
            result = cur.fetchall()
        except Exception as e:
            logger.error(f"Error executing query on database: {e}")
        finally:
            cur.close()
            conn.close()
    return result


def get_conn() -> sqlite3.Connection:
    """
    Creates and returns the connection to the database
    """
    db_path = DB_PATH_NATURAL if DB_NATURAL else DB_PATH_ABBREVIATED

    conn = sqlite3.connect(f'{db_path}')

    _check_connection(conn)

    return conn


def _check_connection(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("PRAGMA database_list")
    except Exception as e:
        logger.error(f"Error connecting to database {DB_NAME}: {e}")
