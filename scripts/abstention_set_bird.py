from sql_metadata import Parser
from collections import Counter
from src.core.extract_instructions import extract_column_table

def extract_gold_sql_db(file_path=".local/train/train/train_gold.sql") -> dict[str, list[str]]:
    """
    Groups gold SQL by their database. Used for later processing.

    Returns: A dictionary of lists. Each dict a database, and each list a list of gold SQL.
    """
    gold_sql = {}

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('--'):
                continue

            tokens = line.split()
            if len(tokens) < 2:
                continue

            db = tokens[-1]
            sql = " ".join(tokens[:-1])

            gold_sql.setdefault(db, []).append(sql)

    return gold_sql


def pool_columns(sql_queries: list[str]) -> Counter:
    """
    Pools all columns for gold sql for a database, used to remove x% of columns later.
    """
    columns_pool = Counter()
    
    for line in sql_queries:
        sql = Parser(line)
        for col in sql.columns:
            if '.' not in col and len(sql.tables) == 1:
                col = sql.tables[0] + '.' + col
            elif '.' not in col:
                print('print')
                for token in sql.tokens:
                    if token.value == col and token.last_keyword.upper() != 'SELECT':
                        def find_table_prev(t):
                            if t.last_keyword == 'FROM' and t.is_name:
                                return t.value
                            find_table_prev(t.previ)

            columns_pool.update([col])

    return columns_pool




if __name__ == '__main__':
    dict = extract_gold_sql_db()
    for key, val in dict.items():
        pool = pool_columns(val)