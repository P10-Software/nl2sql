import os
import psycopg2
from dotenv import load_dotenv
from src.common.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

USER = os.getenv('PG_USER')
PASSWORD = os.getenv('PG_PASSWORD')
HOST = os.getenv('PG_HOST')
PORT = os.getenv('PG_PORT')
DB_NAME = os.getenv('DB_NAME')


def execute_query(query: str):
    """
    Executes a query on the database and returns the raw answer

    Args:
    - query: SQL query as a string.

    Returns:
    - The result of executing the SQL.
    """

    result = None
    conn = create_psql_connection()
    if conn:
        try:
            conn.autocommit = True
            cur = conn.cursor()

            cur.execute(query)
            result = cur.fetchall()

            cur.close()
            conn.close()
        except Exception as e:
            logger.error(f"Error executing query on database: {e}")

    return result


def create_psql_connection():
    """
    Creates and returns the connection to the database
    """
    conn = None

    try:
        conn = psycopg2.connect(dbname=DB_NAME, user=USER, password=PASSWORD, host=HOST, port=PORT)
    except Exception as e:
        logger.error(f"Error connecting to database {DB_NAME}: {e}")

    return conn
