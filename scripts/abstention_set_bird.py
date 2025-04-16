from sql_metadata import Parser
from collections import Counter
from typing import Literal, List, Dict
from json import dump, load
from sqlalchemy import create_engine
from mschema import schema_engine
from tqdm import tqdm
import os
import sqlite3


SEARCH_DIRECTION = Literal['front', 'back']


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


def extract_gold_columns(sql_query: str) -> List[str]:
    """
    Extracts the needed columns of an SQL query.
    """
    sql = Parser(sql_query)
    cols = set()
    for col in sql.columns:
        col = _qualify_column(col, sql)
        cols.add(col)
    return list(cols)


def extract_m_schemas(db_paths):
    """
    Extracts the m-schema of a database.

    Args:
    - db_path: A dictionary where the db name is the key, and its file path the value.
    """
    m_schemas = {}
    for db_id, db_path in tqdm(db_paths.items(), desc="mschemas"):
        db_engine = create_engine(f'sqlite:///{db_path}')
        m_schemas[db_id] = schema_engine.SchemaEngine(engine=db_engine, db_name=db_id).mschema.to_mschema()
    return m_schemas             


def select_columns_for_removal(sql_distribution: Dict, percentage_removal: int):
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
    db_paths = _list_databases()
    columns_to_remove = {}
    for db, col_dist in sql_distribution.items():
        sorted_cols = sorted(col_dist.items(), key=lambda x: abs(x[1] - percentage_removal))
        selected = []
        cumulative = 0

        for i, (col, percent) in enumerate(sorted_cols):
            if percent > percentage_removal:
                continue
            t, c = col.split('.', 1)
            if not _is_safe_to_drop(db_paths[db], t, c):
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
            if '.' not in col:
                continue
            table, column = col.split('.', 1)
            if not _is_safe_to_drop(db_paths[db], table, column):
                continue
            selected.append(col)
            cumulative += percent

        columns_to_remove[db] = selected

    return columns_to_remove


def create_labelled_training_set(removable_columns: Dict, sql_queries: Dict, bird_train_locale: str = ".local/train/train/train.json"):
    """
    Creates the training set with labels.
    Uses training set from bird to create new abstention training set.

    Args:
    - removable_columns (dict): A Dictioanry containing which columns should be removed from which database.
    - sql_queries (dict): A Dictionary containing sql queries and their database.
    - bird_train_locale (str): Path to location of BIRD tain.json file.

    Returns:
    - A training set in dictionary form containing:
        - db_id: name of the database
        - question: The NL question
        - feasible: wether the question is answerable, 0 for no, 1 for yes.
    """
    feasible, infeasible = _label_queries(removable_columns, sql_queries)
    db_paths = _list_databases()
    m_schemas = extract_m_schemas(db_paths)
    with open(bird_train_locale, 'r') as fp:
        bird_train_set = load(fp)

    bird_abstention_set = []

    for val in tqdm(bird_train_set, desc="Building training set, on val:"):
        question = {
            'db_id': val.get('db_id'),
            'question': val.get('question'),
            'SQL': val.get('SQL')
        }
        if val.get('SQL') in infeasible:
            question['feasible'] = 0
            question['columns'] = []
        elif val.get('SQL') in feasible:
            question['feasible'] = 1
            question['columns'] = extract_gold_columns(val.get('SQL'))
        else:
            question['feasible'] = None
        question['schema'] = m_schemas[val.get('db_id')]
        bird_abstention_set.append(question)

    return bird_abstention_set


def remove_cols_from_databases(removable_columns):
    database_paths = _list_databases()

    for db, cols in removable_columns.items():
        table_to_cols = {}
        for col in cols:
            if '.' not in col or 'T' in col or 'None' in col:
                continue
            table, column = col.split('.', 1)
            table_to_cols.setdefault(table, []).append(column)

        db_path = database_paths.get(db)
        if not db_path:
            print(f"Database path not found for {db}")
            continue

        conn = sqlite3.connect(db_path)
        c = conn.cursor()

        for table, columns in table_to_cols.items():
            for column in columns:
                try:
                    sql = f'ALTER TABLE "{table}" DROP COLUMN "{column}";'
                    c.execute(sql)
                    print(f"Dropped column {column} from {table} in {db}")
                except sqlite3.OperationalError as e:
                    print(f"Failed to drop {column} from {table} in {db}: {e}")

        conn.commit()
        conn.close()


def _is_safe_to_drop(db_str, table, column) -> bool:
    conn = sqlite3.connect(db_str)
    cursor = conn.cursor()

    try:
        cursor.execute(f'PRAGMA table_info("{table}");')
        table_info = cursor.fetchall()
    except sqlite3.OperationalError as e:
        print(f"[WARNING] Could not inspect table '{table}': {e}")
        conn.close()
        return False

    all_columns = [row[1] for row in table_info]
    pk_columns = [row[1] for row in table_info if row[5]]  # row[5] = pk flag

    try:
        cursor.execute(f'PRAGMA foreign_key_list("{table}");')
        fk_columns = [row[3] for row in cursor.fetchall()]  # row[3] = 'from' column
    except sqlite3.OperationalError as e:
        print(f"[WARNING] Could not get foreign keys for table '{table}': {e}")
        fk_columns = []

    conn.close()

    return (
        column in all_columns and
        column not in pk_columns and
        column not in fk_columns
    )


def _list_databases(bird_train_locale=".local/train/train/train_databases/train_databases"):
    root_dir = os.path.dirname(os.path.abspath(bird_train_locale))
    db_paths = {}

    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames = [d for d in dirnames if not d.startswith('.') and not d.startswith('__')]

        for filename in filenames:
            if filename.startswith('.') or filename.startswith('__'):
                continue
            if filename.endswith(".sqlite"):
                name = os.path.splitext(filename)[0]
                full_path = os.path.join(dirpath, filename)
                db_paths[name] = full_path

    return db_paths


def _label_queries(removable_columns, sql_queries):
    infeasible = []
    feasible = []
    for db, queries in sql_queries.items():
        columns_to_remove = set(removable_columns.get(db, []))
        for query in queries:
            sql = Parser(query)
            used_columns = {_qualify_column(col, sql) for col in sql.columns}
            if used_columns & columns_to_remove:
                infeasible.append(query)
            else:
                feasible.append(query)

    return feasible, infeasible


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
    infeasible_sql_percent = 15

    gold_sql_dict = extract_gold_sql_db()

    column_count_dict = {}
    for key, val in gold_sql_dict.items():
        pool = pool_columns(val)
        column_count_dict[key] = pool

    cols_to_del = select_columns_for_removal(column_count_dict, infeasible_sql_percent)

    remove_cols_from_databases(cols_to_del)

    new_dataset = create_labelled_training_set(cols_to_del, gold_sql_dict)

    with open(".local/bird_abstention_train_set.json", 'w') as fp:
        dump(new_dataset, fp, indent=4)
