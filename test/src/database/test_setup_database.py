import pytest
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

USER = os.getenv('PG_USER')
PASSWORD = os.getenv('PG_PASSWORD')
HOST = os.getenv('PG_HOST')
PORT = os.getenv('PG_PORT')
DB_NAME = os.getenv('DB_NAME')

class TestSetupDatabase:
    def test_db_exists(self):
        conn = psycopg2.connect(dbname="postgres", user=USER, password=PASSWORD, host=HOST, port=PORT)
        cur = conn.cursor()

        cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{DB_NAME}'")
        exists = cur.fetchone()

        cur.close()
        conn.close()

        assert exists


    def test_if_tables_loaded(self):
        conn = psycopg2.connect(dbname=DB_NAME, user=USER, password=PASSWORD, host=HOST, port=PORT)
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM column_label_lookup")
        res = cur.fetchone()

        if res is not None:
            res = res[0]
        else:
            res = 0

        cur.close()
        conn.close()

        assert res != 0
