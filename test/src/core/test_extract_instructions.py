import sqlite3
from unittest.mock import patch
import pytest
import os
from src.core.extract_instructions import get_query_build_instruct, sanitise_query, extract_column_table

@pytest.fixture
def create_mock_database_file():
    if not os.path.exists("test/src/core/temp/"):
        os.makedirs("test/src/core/temp/")

    db_path = "test/src/core/temp/mock_db.sqlite"
    if os.path.exists(f"{db_path}"):
            os.remove(f"{db_path}")
    conn = sqlite3.connect(db_path)

    build_intstructions = """
    CREATE TABLE users (
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL
    );

    CREATE TABLE orders (
        order_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        amount REAL NOT NULL
    );

    INSERT INTO users (name, email) VALUES
        ('Alice Johnson', 'alice@example.com'),
        ('Bob Smith', 'bob@example.com');

    INSERT INTO orders (user_id, amount) VALUES
        (1, 99.99),
        (2, 149.50),
        (1, 200.00);

    """
    conn.executescript(build_intstructions)
    conn.commit()
    conn.close()

@pytest.mark.parametrize("kind, expected", [
    ('Columns', "CREATE TABLE tab_pln (pln_id TEXT NOT NULL);"),
    ('Tables', "CREATE TABLE tab_pln (pln_id TEXT NOT NULL,\n    pln_name TEXT,\n    created_at DATE DEFAULT CURRENT_TIMESTAMP);"),
    ('Full', "CREATE TABLE tab_pln (pln_id TEXT NOT NULL,\n    pln_name TEXT,\n    created_at DATE DEFAULT CURRENT_TIMESTAMP);\n\nCREATE TABLE nu_pln (nu_id TEXT NOT NULL);")
])
@patch('src.core.extract_instructions._create_build_instruction_tree')
def test_get_query_build_instructions(mock_build_tree, kind, expected):
    # Arrange
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
    result = get_query_build_instruct(kind, query)

    # Assert
    assert result.strip() == expected.strip()


@pytest.mark.parametrize("sql, expected", [
    ("SELECT name FROM names WHERE name LIKE '%john%';",
     "SELECT name FROM names WHERE name LIKE '';"),
    ("SELECT name, age FROM people WHERE name like '%Johnny%' AND age > 18;",
     "SELECT name, age FROM people WHERE name like '' AND age > 18;"),
    ("SELECT age FROM people;", "SELECT age FROM people;"),
    ("SELECT COUNT(name) FROM people;", "SELECT name FROM people;"),
    ("SELECT COUNT(name) FROM people where age > AVG(age);", "SELECT name FROM people where age > age;")
])
def test_sanitise_query_wo_db_access(sql, expected):
    # Arrange + Act
    result = sanitise_query(sql)

    # Assert
    assert result == expected

@pytest.mark.parametrize("sql, expected", [
    ("SELECT COUNT(*) FROM users;", "SELECT user_id, name, email FROM users;"),
    ("SELECT name, COUNT(*) FROM users;", "SELECT name, * FROM users;")
])
def test_sanitise_query_w_db_access(create_mock_database_file, sql, expected):
    # Arrange + Act
    result = sanitise_query(sql, "test/src/core/temp/mock_db.sqlite")

    # Assert
    assert result == expected    


@pytest.mark.parametrize("sql, expected", [
    ("SELECT name, age FROM people WHERE name LIKE '% s.c master%'", {'people': ['name', 'age']}),
    ("SELECT age FROM people", {'people': ['age']}),
    ("SELECT age, name FROM people AND SELECT gender FROM genders", {'people': ['age', 'name'], 'genders': ['gender']}),
    ("SELECT name FROM people and SELECT gender FROM people WHERE gender LIKE '% SELECT age FROM people%'", {'people': ['name', 'gender']}),
    ("SELECT people.name FROM people and SELECT genders.gender FROM genders", {'people': ['name'], 'genders': ['gender']}),
    ("SELECT people.name FROM people and SELECT gender FROM genders", {'people': ['name'], 'genders': ['gender']}),
    ("SELECT T1.name from people as T1", {'people': ['name']}),
    ("SELECT name, * FROM people;", {'people': ['name']}) # * should be ignored
])
def test_extract_column_table(sql, expected):
    # Arrange + Act
    result = extract_column_table(sql)

    # Assert
    assert result == expected


@pytest.mark.parametrize("kind, expected", [
    ('Full', "CREATE TABLE tab_pln (pln_id TEXT NOT NULL,\n    pln_name TEXT,\n    created_at DATE DEFAULT CURRENT_TIMESTAMP);\n\nCREATE TABLE nu_pln (nu_id TEXT NOT NULL);"),
    ('Table', "CREATE TABLE tab_pln (pln_id TEXT NOT NULL,\n    pln_name TEXT,\n    created_at DATE DEFAULT CURRENT_TIMESTAMP);\n\nCREATE TABLE nu_pln (nu_id TEXT NOT NULL);"),
    ('column', "CREATE TABLE tab_pln (pln_id TEXT NOT NULL,\n    pln_name TEXT,\n    created_at DATE DEFAULT CURRENT_TIMESTAMP);\n\nCREATE TABLE nu_pln (nu_id TEXT NOT NULL);")
])
@patch('src.core.extract_instructions._create_build_instruction_tree')
def test_none_query(mock_build_tree, kind, expected):
    # Arrange
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
    sql = None

    # Act
    res = get_query_build_instruct(kind, sql)

    # Assert
    assert res == expected
