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
        cols = set()
        sql = Parser(line)
        for col in sql.columns:
            col = _qualify_column(col, sql)
            cols.add(col)
        columns_pool.update(cols)

    for col in columns_pool:
        columns_pool[col] = round((columns_pool[col] / len(sql_queries)) * 100, 2)

    return Counter(dict(columns_pool.most_common()))  # Sorts the counter


def select_columns_for_removal(sql_distribution, percentage_removal: int):
    """
    Selects columns to remove from a database in order to make approximately [percentage_removal]% 
    of the gold SQL queries infeasible. If an exact match is not possible, the function selects as 
    close a percentage as possible without exceeding the target.

    Args:
    - sql_distribution (dict): A dictionary mapping each column to the number or percentage of 
      gold queries in which it appears.
    - percentage_removal (int): Target percentage of gold queries to make infeasible by removing columns.

    Returns:
    - A dictionary specifying:
        - The columns (grouped by database) to be removed.
    """
    columns_to_remove = {}
    for db, col_dist in sql_distribution.items():
        sorted_cols = sorted(col_dist.items(), key=lambda x: abs(x[1] - percentage_removal))
        selected = []
        cumulative = 0

        for i, (col, percent) in enumerate(sorted_cols):
            if percent > percentage_removal:
                continue
            selected.append(col)
            cumulative += percent
            break

        remaining = sorted(
            [(col, p) for col, p in col_dist.items() if col not in selected],
            key=lambda x: -x[1]
        )

        for col, percent in remaining:
            if cumulative + percent > percentage_removal:
                continue
            selected.append(col)
            cumulative += percent

        columns_to_remove[db] = selected

    return columns_to_remove


def build_training_set():
    pass


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

    cols_to_del = select_columns_for_removal(column_count_dict, 15)

    with open(".local/BirdBertTrain.json", 'w') as fp:
        dump(column_count_dict, fp, indent=5)
