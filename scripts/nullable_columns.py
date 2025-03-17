import os
import sqlite3
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DB_NAME = os.getenv('DB_NAME')
DB_PATH_ABBREVIATED = os.getenv('DB_PATH_ABBREVIATED')
DB_PATH_NATURAL = os.getenv('DB_PATH_NATURAL')
DB_NATURAL = int(os.getenv('DB_NATURAL', 0))


#TODO Remember to add one for both abbreviated and natural (Just add to the same)

def find_nullable_columns(db_path, nullable_columns):
    """
    Identify columns that are nullable if they contain any rows containing nulle values.
    """

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    for table in tables:
        table_name = table[0]

        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()

        for column in columns:
            column_name = column[1]

            # Query returns 1 if where exist a row with null else 0
            query = f"SELECT EXISTS(SELECT 1 FROM {table_name} WHERE {column_name} IS NULL LIMIT 1);"
            cursor.execute(query)
            nullable = cursor.fetchall()[0][0]
            full_column_name = f"{table_name}.{column_name}"
            nullable_columns.append({"table.column": full_column_name, "nullable": nullable})

    conn.close()

    return nullable_columns


def save_nullable_columns(nullable_columns):
    """
    Save the nullable_column in a CSV format.
    """
    df = pd.DataFrame(nullable_columns)

    csv_file = ".local/nullable_columns.csv"
    df.to_csv(csv_file, index=False)

    print(f"Nullable columns saved to {csv_file}")


if __name__ == "__main__":
    data = []

    data = find_nullable_columns(DB_PATH_ABBREVIATED, data)
    data = find_nullable_columns(DB_PATH_NATURAL, data)

    save_nullable_columns(data)
