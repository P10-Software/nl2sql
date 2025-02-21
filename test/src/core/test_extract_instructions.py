import sqlite3
from unittest.mock import patch
import pytest
import pandas as pd
from src.core.extract_instructions import get_query_build_instruct, _sanitise_query, _extract_column_table, _transform_natural_query


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


@pytest.mark.parametrize("kind, expected", [
    ('Columns', "CREATE TABLE tab_pln (pln_id TEXT NOT NULL);"),
    ('Tables', "CREATE TABLE tab_pln (pln_id TEXT NOT NULL,\n    pln_name TEXT,\n    created_at DATE DEFAULT CURRENT_TIMESTAMP);"),
    ('Full', "CREATE TABLE tab_pln (pln_id TEXT NOT NULL,\n    pln_name TEXT,\n    created_at DATE DEFAULT CURRENT_TIMESTAMP);\n\nCREATE TABLE nu_pln (nu_id TEXT NOT NULL);")
])
@patch('src.core.extract_instructions.get_conn')
@patch('src.core.extract_instructions._create_build_instruction_tree')
def test_get_query_build_instructions(mock_build_tree, mock_get_conn, in_memory_db, kind, expected):
    # Arrange
    mock_get_conn.return_value = in_memory_db
    mock_build_tree.return_value = {
        'tab_pln': {
            'create_table': 'CREATE TABLE tab_pln ({columns});',
            'columns': {
                'pln_id': 'pln_id TEXT NOT NULL',
                'pln_name': 'pln_name TEXT',
                'created_at': 'created_at DATE DEFAULT CURRENT_TIMESTAMP'
            }
        },
        'nu_pln': {
            'create_table': 'CREATE TABLE nu_pln ({columns});',
            'columns': {
                'nu_id': 'nu_id TEXT NOT NULL'
            }
        }
    }
    query = "SELECT pln_id FROM tab_pln"

    # Act
    result = get_query_build_instruct(kind, query, False)

    # Assert
    assert result.strip() == expected.strip()


@pytest.mark.parametrize("kind, translated_column_table, expected", [
    ('Columns', {'table_plan': ['plan_id']}, "CREATE TABLE table_plan (plan_id TEXT NOT NULL);"),
    ('Tables', {'table_plan': ['plan_id', 'plan_name', 'created_at']}, "CREATE TABLE table_plan (plan_id TEXT NOT NULL,\n    plan_name TEXT,\n    created_at DATE DEFAULT CURRENT_TIMESTAMP);"),
    ('Full', {'table_plan': ['plan_id', 'plan_name', 'created_at'], 'nu_plan': ['nu_id']}, "CREATE TABLE table_plan (plan_id TEXT NOT NULL,\n    plan_name TEXT,\n    created_at DATE DEFAULT CURRENT_TIMESTAMP);\n\nCREATE TABLE nu_plan (nu_id TEXT NOT NULL);")
])
@patch("src.core.extract_instructions._transform_natural_query")
@patch('src.core.extract_instructions.get_conn')
@patch('src.core.extract_instructions._create_build_instruction_tree')
def test_get_query_build_instructions_naturalised(mock_build_tree, mock_get_conn, mock_transform_natural_query, in_memory_db, kind, translated_column_table, expected):
    # Arrange
    mock_get_conn.return_value = in_memory_db
    mock_build_tree.return_value = {
        'table_plan': {
            'create_table': 'CREATE TABLE table_plan ({columns});',
            'columns': {
                'plan_id': 'plan_id TEXT NOT NULL',
                'plan_name': 'plan_name TEXT',
                'created_at': 'created_at DATE DEFAULT CURRENT_TIMESTAMP'
            }
        },
        'nu_plan': {
            'create_table': 'CREATE TABLE nu_plan ({columns});',
            'columns': {
                'nu_id': 'nu_id TEXT NOT NULL'
            }
        }
    }

    mock_transform_natural_query.return_value = translated_column_table

    query = "SELECT pln_id FROM tab_pln"

    # Act
    result = get_query_build_instruct(kind, query, True)

    print("TEST: ", result)

    # Assert
    assert result.strip() == expected.strip()


@patch("src.core.extract_instructions.pd.read_csv")
def test_transform_natural_query(mock_read_csv):

    mock_table_names = pd.DataFrame({
        'old_name': ['tab_pln', 'trl_tp'],
        'new_name': ['table_plan', 'trial_type']
    })

    mock_column_names = pd.DataFrame({
        'old_name': ['trl_id', 'pln_id', 'pln_name'],
        'new_name': ['trial_id', 'plan_id', 'plan_name']
    })

    mock_read_csv.side_effect = [mock_table_names, mock_column_names]

    selected_tables_columns = {'tab_pln': ['pln_id', 'pln_name']}

    expected = {'table_plan': ['plan_id', 'plan_name']}

    result = _transform_natural_query(selected_tables_columns)

    assert result == expected


@pytest.mark.parametrize("sql, expected", [
    ("SELECT name FROM names WHERE name LIKE '%john%';",
     "SELECT name FROM names WHERE name LIKE '';"),
    ("SELECT name, age FROM people WHERE name like '%Johnny%' AND age > 18;",
     "SELECT name, age FROM people WHERE name like '' AND age > 18;"),
    ("SELECT age FROM people;", "SELECT age FROM people;")
])
def test_sanitise_query(sql, expected):
    # Arrange + Act
    result = _sanitise_query(sql)

    # Assert
    assert result == expected


@pytest.mark.parametrize("sql, expected", [
    ("SELECT name, age FROM people WHERE name LIKE '% s.c master%'", {'people': ['name', 'age']}),
    ("SELECT age FROM people", {'people': ['age']}),
    ("SELECT age, name FROM people AND SELECT gender FROM genders", {'people': ['age', 'name'], 'genders': ['gender']}),
    ("SELECT name FROM people and SELECT gender FROM people WHERE gender LIKE '% SELECT age FROM people%'", {'people': ['name', 'gender']}),
    ("SELECT people.name FROM people and SELECT genders.gender FROM genders", {'people': ['name'], 'genders': ['gender']}),
    ("SELECT people.name FROM people and SELECT gender FROM genders", {'people': ['name'], 'genders': ['gender']}),
    ("SELECT T1.name from people as T1", {'people': ['name']})
])
def test_extract_column_table(sql, expected):
    # Arrange + Act
    result = _extract_column_table(sql)

    # Assert
    assert result == expected
