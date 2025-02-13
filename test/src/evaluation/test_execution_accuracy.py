import sqlite3
import pytest
from unittest.mock import patch
from src.evaluation.execution_accuracy import execution_accuracy
from src.database.database import execute_query 


@pytest.fixture(scope="function")
def in_memory_db():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE tab_pln (
            pln_id TEXT NOT NULL,
            pln_name TEXT,
            created_at DATE DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE nu_pln (
            nu_id TEXT NOT NULL
        )
    """)
    cursor.execute("INSERT INTO tab_pln (pln_id, pln_name) VALUES ('id1', 'BigJoel')")
    cursor.execute("INSERT INTO tab_pln (pln_id, pln_name) VALUES ('id2', 'SmallJoel')")
    cursor.execute("INSERT INTO tab_pln (pln_id, pln_name) VALUES ('id3', 'NormalJoel')")
    cursor.execute("INSERT INTO nu_pln (nu_id) VALUES ('id1')")
    cursor.execute("INSERT INTO nu_pln (nu_id) VALUES ('id2')")
    conn.commit()
    yield conn
    conn.close()


@pytest.mark.parametrize("generated_query, goal_query, expected", [
    ("SELECT * FROM tab_pln", "SELECT * FROM tab_pln", 1)
])
@patch('src.database.evaluation.res_goal_query')
@patch('src.database.evaluation.res_generated_query')
def test_ex(mock_res_generated_query, mock_res_goal_query, in_memory_db, goal_query, generated_query, expected):
    pass
    # TODO Consider how to handle it? the db is in execute_query... Maybe just use execute_query twice here.
