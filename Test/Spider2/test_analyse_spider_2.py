from Src.Spider2Lite.analyze_spider_2 import get_columns_from_table_json_file, analyze_sqlite_database, analyze_all_sqlite_databases
import pytest
import os
from os.path import join
from json import dump

PATH_TO_MOCK_DB_DIR = "Test/Spider2/mock_databases/sqlite"
PATH_TO_MOCK_DB1 = join(PATH_TO_MOCK_DB_DIR, "db1")
PATH_TO_MOCK_DB2 = join(PATH_TO_MOCK_DB_DIR, "db2")

class TestAnalyzeSpider2():
    @pytest.fixture
    def setup_mock_db_and_table_content(self):
        if not os.path.exists(PATH_TO_MOCK_DB_DIR):
            os.makedirs(PATH_TO_MOCK_DB1)
            os.makedirs(PATH_TO_MOCK_DB2)

            #Make table for mock db 1
            with open(join(PATH_TO_MOCK_DB1, "table1.json"), "w") as file:
                dump({"descriptions": ["test1", "test2"], "column_names": ["column1", "column2", "column3"]}, file, indent=4)


            # Make tables for mock db 2
            with open(join(PATH_TO_MOCK_DB2, "table1.json"), "w") as file:
                dump({"descriptions": ["test1", "test2"], "column_names": ["column1", "column2", "column3"]}, file, indent=4)

            with open(join(PATH_TO_MOCK_DB2, "table2.json"), "w") as file:
                dump({"descriptions": ["test1", "test3", "test4", "test5"], "column_names": ["column1", "column3", "column4", "column5"]}, file, indent=4)

    @pytest.mark.parametrize("table_path, expected_columns", [
        (join(PATH_TO_MOCK_DB1, "table1.json"), ["column1", "column2", "column3"]),
        (join(PATH_TO_MOCK_DB2, "table1.json"), ["column1", "column2", "column3"]),
        (join(PATH_TO_MOCK_DB2, "table2.json"), ["column1", "column3", "column4", "column5"])
    ])
    def test_get_columns_from_table_json_file(self, setup_mock_db_and_table_content, table_path, expected_columns):
        actual_columns = get_columns_from_table_json_file(table_path)
        assert(actual_columns == expected_columns)

    @pytest.mark.parametrize("db_path, expected_result", [
        (PATH_TO_MOCK_DB1, (1, 3, 0)), #The database db1 has only 1 table with 3 columns - hence no duplicates exist
        (PATH_TO_MOCK_DB2, (2, 7, 2))  #The database db2 has 2 tables with 7 columns with 2 of these shared between the tables
    ])
    def test_analyze_sqlite_database(self, setup_mock_db_and_table_content, db_path, expected_result):
        actual_result = analyze_sqlite_database(db_path)
        assert(actual_result == expected_result)

    def test_analyze_all_sqlite_databases(self, setup_mock_db_and_table_content):
        # Arrange
        expected_overall_statistics = {
            "number of databases": 2,
            "total number of tables": 3,
            "total number of columns": 10,
            "total number of duplicate columns": 2,
            "average number of tables per db": 1.5,
            "average number of columns per db": 5,
            "average number of columns per table": 10 / 3
        }
        expected_statistics_for_db1 = {"number of tables": 1, "number of columns": 3, "number of duplicate columns": 0}
        expected_statistics_for_db2 = {"number of tables": 2, "number of columns": 7, "number of duplicate columns": 2}

        #Act
        result = analyze_all_sqlite_databases(PATH_TO_MOCK_DB_DIR)

        #Assert
        assert(result["overall statistics"] == expected_overall_statistics)
        assert(result[PATH_TO_MOCK_DB1] == expected_statistics_for_db1)
        assert(result[PATH_TO_MOCK_DB2] == expected_statistics_for_db2)