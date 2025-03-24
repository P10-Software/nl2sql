import pytest
import sqlite3
from unittest.mock import MagicMock, patch
from src.database.database import verify_database


@pytest.fixture
def in_memory_db():
    conn = sqlite3.connect(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def mock_logger():
    with patch('src.database.database.logger') as mock_logger:
        yield mock_logger


def test_empty_database(in_memory_db, mock_logger):
    # Arrange, Act & Assert
    assert not verify_database(in_memory_db)
    mock_logger.error.assert_called_with(
        "No tables found in the database. Please check database name and setup. Quitting...")


def test_database_all_empty_table(in_memory_db, mock_logger):
    # Arrange
    cursor = in_memory_db.cursor()
    cursor.execute("CREATE TABLE test (id INTEGER, value TEXT)")
    in_memory_db.commit()

    # Act + Assert
    assert not verify_database(in_memory_db)
    mock_logger.error.assert_called_with("1 tables found, but all were empty.")


def test_database_with_data(in_memory_db, mock_logger):
    # Arrange
    cursor = in_memory_db.cursor()
    cursor.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT);")
    cursor.execute("INSERT INTO test (value) VALUES ('sample data');")
    in_memory_db.commit()

    # Act + Assert
    assert verify_database(in_memory_db)
    mock_logger.info.assert_called_with("Found data in 1 out of 1.")


def test_sqlite_error(mock_logger):
    # Arrange
    mock_conn = MagicMock()
    mock_conn.cursor.side_effect = sqlite3.Error("mock error")

    # Act + Assert
    assert not verify_database(mock_conn)
    mock_logger.error.assert_called_with("SQLite error mock error")
