import scripts.create_training_set_from_spider
scripts.create_training_set_from_spider.PATH_TO_SPIDER_DIR = "test/scripts/mock_dataset"

from scripts.create_training_set_from_spider import _load_schema_for_all_dbs, create_training_set, _lowercase_column_and_table_names, m_schema_to_ddl
import os
import sqlite3
import pytest
import json
import shutil

@pytest.fixture
def create_mock_database_folder():
    os.makedirs("test/scripts/mock_dataset/database/mock_db")
    open("test/scripts/mock_dataset/database/mock_db/schema.sql", "w").close()

    db_path = "test/scripts/mock_dataset/database/mock_db/mock_db.sqlite"
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

    yield

    shutil.rmtree('test/scripts/mock_dataset/database/mock_db')
    
@pytest.fixture
def mock_train_file():
    train_content = [{"db_id": "mock_db", "query": "SELECT COUNT(*) FROM users;", "question": "How many users exist?"}]
    with open("test/scripts/mock_dataset/train_spider.json", "w") as file:
        json.dump(train_content, file)

def test_load_schema_for_all_dbs(create_mock_database_folder):
    # arrange
    expected_schema_dict = {
        "mock_db": {"schema": """
【DB_ID】 mock_db
【Schema】
# Table: users
[
(user_id:INTEGER, Primary Key, Examples: [1, 2]),
(name:TEXT, Examples: [Alice Johnson, Bob Smith]),
(email:TEXT)
]
# Table: orders
[
(order_id:INTEGER, Primary Key, Examples: [1, 2, 3]),
(user_id:INTEGER, Examples: [1, 2]),
(amount:REAL, Examples: [99.99, 149.5, 200.0])
]
""", "db_path": "test/scripts/mock_dataset/database/mock_db/mock_db.sqlite"}
    }

    # act
    actual_schema_dict = _load_schema_for_all_dbs()

    # assert
    assert expected_schema_dict.keys() == actual_schema_dict.keys()
    assert set("".join(expected_schema_dict["mock_db"]["schema"].split()).split("#")) == set("".join(actual_schema_dict["mock_db"]["schema"].split()).split("#"))
    assert expected_schema_dict["mock_db"]["db_path"] == actual_schema_dict["mock_db"]["db_path"]

def test_create_training_set(create_mock_database_folder, mock_train_file):
    # arrange
    expected_db = "【DB_ID】 mock_db"
    expected_table_orders = """# Table: orders
[
(order_id:INTEGER, Primary Key, Examples: [1, 2, 3]),
(user_id:INTEGER, Examples: [1, 2]),
(amount:REAL, Examples: [99.99, 149.5, 200.0])
]"""
    expected_table_users = """# Table: users
[
(user_id:INTEGER, Primary Key, Examples: [1, 2]),
(name:TEXT, Examples: [Alice Johnson, Bob Smith]),
(email:TEXT)
]"""
    expected_question = "How many users exist?"
    expected_goal_answer = ["users user_id", "users name", "users email"]

    # act
    actual_training_set, _ = create_training_set()

    # assert
    assert actual_training_set[0]["goal answer"] == expected_goal_answer
    assert expected_db in actual_training_set[0]["schema"]
    assert expected_table_orders in actual_training_set[0]["schema"]
    assert expected_table_users in actual_training_set[0]["schema"]
    assert expected_question == actual_training_set[0]["question"]

def test_lowercase_column_and_table_names():
    # arrange
    input_schema = """
【DB_ID】 mock_db
【Schema】
# Table: Users
[
(user_id:INTEGER, Primary Key, Examples: [1, 2]),
(name:TEXT, Examples: [Alice Johnson, Bob Smith]),
(email:TEXT)
]
# Table: orders
[
(Order_id:INTEGER, Primary Key, Examples: [1, 2, 3]),
(useR_id:INTEGER, Examples: [1, 2]),
(Amount:REAL, Examples: [99.99, 149.5, 200.0])
]
【Foreign keys】
Users.user_id=orders.useR_id
"""
    expected_lowered_schema = """
【DB_ID】 mock_db
【Schema】
# Table: users
[
(user_id:INTEGER, Primary Key, Examples: [1, 2]),
(name:TEXT, Examples: [Alice Johnson, Bob Smith]),
(email:TEXT)
]
# Table: orders
[
(order_id:INTEGER, Primary Key, Examples: [1, 2, 3]),
(user_id:INTEGER, Examples: [1, 2]),
(amount:REAL, Examples: [99.99, 149.5, 200.0])
]
【Foreign keys】
users.user_id=orders.user_id
"""

    # act
    actual_lowered_schema = _lowercase_column_and_table_names(input_schema)

    # assert
    assert expected_lowered_schema == actual_lowered_schema