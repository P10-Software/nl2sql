from json import load, dump
from os import listdir
from os.path import isfile, join
from collections import Counter

def analyze_all_sqlite_databases(path_to_sqlite_dir: str):
    """
    Analyzes all SQLite database files in a given directory.

    This function iterates over all SQLite databases in the specified directory,
    extracting statistics for each database, including the number of tables, columns, 
    and duplicate columns. It also computes overall statistics, such as total and 
    average values across all databases.

    Args:
        path_to_sqlite_dir (str): The path to the directory containing SQLite database files.

    Returns:
        dict: A dictionary containing:
            - "overall statistics": A summary of all databases, including:
                - "number of databases" (int): Total databases analyzed.
                - "total number of tables" (int): Total count of tables across all databases.
                - "total number of columns" (int): Total count of columns across all databases.
                - "total number of duplicate columns" (int): Count of columns appearing in multiple tables.
                - "average number of tables per db" (float): Avg. number of tables per database.
                - "average number of columns per db" (float): Avg. number of columns per database.
                - "average number of columns per table" (float): Avg. number of columns per table.
            - Individual database paths as keys, each mapping to a dictionary containing:
                - "number of tables" (int): Number of tables in the database.
                - "number of columns" (int): Number of columns in the database.
                - "number_of_duplicate_columns" (int): Count of duplicate columns in the database.

    Raises:
        ZeroDivisionError: If there are no databases in the specified directory.
    """
    db_paths = [join(path_to_sqlite_dir, db_path) for db_path in listdir(path_to_sqlite_dir)]

    db_statistics = dict()
    total_number_of_tables = 0
    total_number_of_columns = 0
    total_number_of_duplicate_columns = 0
    for db_path in db_paths:
        number_of_tables, number_of_columns, number_of_duplicate_columns = analyze_sqlite_database(db_path)
        db_statistics[db_path] = {"number of tables": number_of_tables, "number of columns": number_of_columns, "number_of_duplicate_columns": number_of_duplicate_columns}
        total_number_of_tables += number_of_tables
        total_number_of_columns += number_of_columns
        total_number_of_duplicate_columns += number_of_duplicate_columns

    overall_statistics = {"overall statistics": {"number of databases": len(db_paths), "total number of tables": total_number_of_tables, "total number of columns": total_number_of_columns, 
                          "total number of duplicate columns": total_number_of_duplicate_columns, "average number of tables per db": total_number_of_tables / len(db_paths),
                          "average number of columns per db": total_number_of_columns / len(db_paths), "average number of columns per table": total_number_of_columns / total_number_of_tables}}

    return overall_statistics | db_statistics


def analyze_sqlite_database(db_path: str):
    """
    Analyzes a database stored as JSON files in a given directory.

    This function scans the specified directory for JSON files representing database tables,
    extracts column names from each table, and calculates the number of tables, total columns,
    and duplicate columns.

    Args:
        db_path (str): The path to the directory containing the JSON table files.

    Returns:
        tuple: A tuple containing:
            - int: The number of table files found.
            - int: The total number of columns across all tables.
            - int: The number of duplicate columns (columns appearing in multiple tables).
    """
    table_paths = [join(db_path, file) for file in listdir(db_path) if isfile(join(db_path, file)) and file.endswith(".json")]
    columns = list()

    # Get all columns (including duplicates)
    for table_path in table_paths:
        columns.extend(get_columns_from_table_json_file(table_path))


    # Number of occurences for each column
    occurences_for_columns = Counter(columns).items()
    duplicate_columns = [column for column, occurences in occurences_for_columns if occurences > 1]

    return len(table_paths), len(columns), len(duplicate_columns)

def get_columns_from_table_json_file(file_path: str):
    """
    Extracts column names from a JSON table file.

    This function reads a JSON file representing a database table and returns the list of column names.

    Args:
        file_path (str): The path to the JSON file containing the table data.

    Returns:
        list: A list of column names found in the JSON file.
    """

    with open(file_path, "r") as file:
        table_dict = load(file)

    return table_dict["column_names"]

if __name__ == "__main__":
    path_to_sqlite_dir = "src/spider2-lite/resource/databases/sqlite"
    statistics = analyze_all_sqlite_databases(path_to_sqlite_dir)

    with open("src/sqlite_statistics.json", "w") as file:
        dump(statistics, file, indent=4)