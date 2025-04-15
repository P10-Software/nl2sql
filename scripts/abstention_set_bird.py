from sql_metadata import Parser
from collections import Counter
from typing import Literal, List
from json import dump


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


def pool_columns(sql_queries: List[str]) -> Counter:
    """
    Pools all columns for gold sql for a database, used to remove x% of columns later.
    """
    columns_pool = Counter()

    for line in sql_queries:
        sql = Parser(line)
        for col in sql.columns:
            col = _qualify_column(col, sql)
            columns_pool.update([col])

    return Counter(dict(columns_pool.most_common()))


def _qualify_column(col: str, sql) -> str:
    if '.' in col:
        return col

    if len(sql.tables) == 1:
        return f"{sql.tables[0]}.{col}"

    return _infer_column_table(col, sql)


def _infer_column_table(col: str, sql) -> str:
    for token in sql.tokens:
        if token.value != col:
            continue

        direction = 'back' if token.last_keyword.upper() != 'SELECT' else 'front'
        table = _find_table_recurs(token, direction)
        return f"{table}.{col}"

    return col


SEARCH_DIRECTION = Literal['front', 'back']


def _find_table_recurs(token, direction: SEARCH_DIRECTION):
    try:
        if token.position == -1:
            return 'None'

        if token.last_keyword.upper() == 'FROM' and token.is_name:
            return token.value

        next_token = token.next_token if direction == 'front' else token.previous_token
    except Exception:
        return 'None'

    return _find_table_recurs(next_token, direction)


if __name__ == '__main__':
    gold_sql_dict = extract_gold_sql_db()

    column_count_dict = {}

    for key, val in gold_sql_dict.items():
        pool = pool_columns(val)
        column_count_dict[key] = pool

    with open(".local/BirdBertTrain.json", 'w') as fp:
        dump(column_count_dict, fp, indent=5)
