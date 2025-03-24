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


def verify_database(conn: sqlite3.Connection) -> bool:
    """
    Verifies that the database has content.
    """
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = cursor.fetchall()

        if not tables:
            logger.error("No tables found in the database. Please check database name and setup. Quitting...")
            return False

        non_empty_tables = 0

        for table in tables:
            cursor.execute(f"SELECT 1 FROM {table} LIMIT 1;")
            if cursor.fetchone():
                non_empty_tables += 1

        if non_empty_tables != 0:
            logger.info(f"Found data in {non_empty_tables} out of {len(tables)}.")
            return True

        logger.error(f"{len(tables)} tables found, but all were empty.")
        return False
    except sqlite3.Error as e:
        logger.error(f"SQLite error {e}")
        return False


def _check_connection(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("PRAGMA database_list")
    except Exception as e:
        logger.error(f"Error connecting to database {DB_NAME}: {e}")
