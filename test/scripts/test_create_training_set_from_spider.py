import scripts.create_training_set_from_spider
scripts.create_training_set_from_spider.PATH_TO_SPIDER_DIR = "test/scripts/mock_dataset"

from scripts.create_training_set_from_spider import _load_ddl_instructions_for_all_dbs, create_training_set
import os
import sqlite3
import pytest
import json

@pytest.fixture
def create_mock_database_folder():
    if not os.path.exists("test/scripts/mock_dataset"):
        os.makedirs("test/scripts/mock_dataset")

    if not os.path.exists("test/scripts/mock_dataset/database/mock_db"):
        os.makedirs("test/scripts/mock_dataset/database/mock_db")


    open("test/scripts/mock_dataset/database/mock_db/schema.sql", "w").close()

    db_path = "test/scripts/mock_dataset/database/mock_db/mock_db.sqlite"
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
    
@pytest.fixture
def mock_train_file():
    train_content = [{"db_id": "mock_db", "query": "SELECT COUNT(*) FROM users;", "question": "How many users exist?"}]
    with open("test/scripts/mock_dataset/train_spider.json", "w") as file:
        json.dump(train_content, file)

def test__load_ddl_instructions_for_all_dbs(create_mock_database_folder):
    # arrange
    expected_ddl_dict = {
        "mock_db": """
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
"""
    }

    # act
    actual_ddl_dict = _load_ddl_instructions_for_all_dbs()

    # assert
    assert expected_ddl_dict.keys() == actual_ddl_dict.keys()
    assert set("".join(expected_ddl_dict["mock_db"].split()).split("#")) == set("".join(actual_ddl_dict["mock_db"].split()).split("#"))

def test_create_training_set(create_mock_database_folder, mock_train_file):
    # arrange
    expected_input = """
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
To answer: How many users exist?
We need columns:
    """
    expected_training_set = [{"input": expected_input, "goal answer": ["users.user_id", "users.name", "users.email"]}]

    # act
    actual_training_set = create_training_set()

    # assert
    assert len(expected_training_set) == len(actual_training_set)
    assert actual_training_set[0]["goal answer"] == expected_training_set[0]["goal answer"]
    assert actual_training_set[0]["input"] == expected_input
