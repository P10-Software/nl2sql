import os
import sqlite3
from dotenv import load_dotenv
from src.common.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

DB_NAME = os.getenv('DB_NAME')
DB_PATH = os.getenv('DB_PATH')


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
    conn = sqlite3.connect(f'{DB_PATH}')
    _check_connection(conn)

    return conn


def _check_connection(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("PRAGMA database_list")
    except Exception as e:
        logger.error(f"Error connecting to database {DB_NAME}: {e}")
