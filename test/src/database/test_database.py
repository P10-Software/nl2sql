import sqlite3
import pytest
from unittest.mock import patch
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
    cursor.execute(
        "INSERT INTO tab_pln (pln_id, pln_name) VALUES ('id111', 'BigJoel')")
    cursor.execute("INSERT INTO nu_pln (nu_id) VALUES ('id1')")
    conn.commit()
    yield conn
    conn.close()


@pytest.mark.parametrize("query, expected_result", [
    ("SELECT pln_name FROM tab_pln", [('BigJoel',)]),
    ("SELECT pln_id, pln_name FROM tab_pln", [('id111', 'BigJoel')])
])
@patch('src.database.database.get_conn')
def test_execute_query(mock_get_conn, in_memory_db, query, expected_result):
    mock_get_conn.return_value = in_memory_db

    res = execute_query(query)

    assert res == expected_result
