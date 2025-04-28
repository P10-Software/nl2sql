from typing import Literal
import sqlite3
import re
import pandas as pd
from sql_metadata import Parser
from src.common.logger import get_logger
from src.database.database import execute_query

SchemaKind = Literal['Full', 'Tables', 'Columns']
logger = get_logger(__name__)


def get_query_build_instruct(kind: SchemaKind, query: str, db_path: str = "") -> str:
    """
    Find the build instructions of the database based on a query.

    Args:
    - kind (SchemaKind): One of 'Full', 'Tables', 'Columns' specifying how restricted the schema should be.
    - query: SQL query in string format
    - db_path: Optional path to sqlite DB

    Returns:
    - SQL build instructions (sql): SQL instructions specifying how to build the DB.
    """

    if query is None or '':
        kind = 'Full'
        query = ''
        selected_tables_columns = None
    else:
        selected_tables_columns = extract_column_table(query, db_path)

    schema_tree = _create_build_instruction_tree()

    return _create_build_instruction(schema_tree, selected_tables_columns, kind)


def extract_column_table(query: str, db_path: str = "", replace_all_with_single: bool = False) -> dict[str, list[str]]:
    column_table_mapping = {}
    sanitised_query = sanitise_query(query, db_path, replace_all_with_single)
    subqueries = re.split(" UNION | INTERSECT | EXCEPT", sanitised_query)
    for subquery in subqueries:
        parser = Parser(subquery)
        tables = parser.tables
        columns = parser.columns

        if len(tables) == 1:
            single_table = tables[0]
            for col in columns:
                if '*' in col:
                    continue
                if '.' in col:
                    table_name, column_name = col.split('.')
                    column_table_mapping.setdefault(
                        table_name, []).append(column_name)
                else:
                    column_table_mapping.setdefault(single_table, []).append(col)
        else:
            for col in columns:
                if '*' in col:
                    continue
                if '.' in col:
                    try:
                        table_name, column_name = col.split('.')
                    except:
                        raise Exception(col)
                    column_table_mapping.setdefault(
                        table_name, []).append(column_name)
                else:
                    logger.error(f"ERROR extracting columns, found ambiguity in query: {query}")
                    raise Exception(f"ERROR extracting columns, found ambiguity in query: {query}")
                
    if not column_table_mapping: 
        logger.error(f"ERROR empty column table mapping")
        raise Exception(f"ERROR empty column table mapping")

    return column_table_mapping


def sanitise_query(query: str, db_path: str = "", replace_all_with_single: bool = False):
    pre_sanitized_query = query
    query = re.sub(r"\"[^\"]*\"", r"''", query, flags=re.IGNORECASE)

    # Replace * with all columns in tables if no other columns are selected
    if db_path:
        # split at FROM keyword - if len > 2 a subquery is present
        subparts = query.split("FROM") 
        if "*" in subparts[0] and not "," in subparts[0]:
            subparts[0] = re.sub(r"(?:\w+\(([\w\*]+)\))", r"\1", subparts[0], flags=re.IGNORECASE)
            query = "FROM".join(subparts)
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                column_names = [desc[0] for desc in cursor.description]
                from_part_tokens = subparts[1].split("WHERE")[0].strip().split(" ")
                if len(from_part_tokens) == 1:
                    table_name = from_part_tokens[0].removesuffix(";")
                    column_names = [f"{table_name}.{column}" for column in column_names]

                if len(column_names) < 1:
                    raise Exception("No column names returned")
                elif len(column_names) == 1 or replace_all_with_single:
                    replacement_string = column_names[0]
                else:
                    replacement_string = ", ".join(column_names)

                subparts[0] = subparts[0].replace("*", replacement_string)
                query = "FROM".join(subparts)
    
        if len(subparts) > 2:
            index_of_current_subquery = 1
            while index_of_current_subquery <= len(subparts) - 2:
                if "*" in subparts[index_of_current_subquery]:
                    logger.warning(f"Query might contain subquery: {pre_sanitized_query}")
                    break
                index_of_current_subquery += 1

    query = re.sub(r"(?:\w+\(([\w\*]+)\))", r"\1", query, flags=re.IGNORECASE)

    return query

def _create_build_instruction_tree() -> dict:
    """
    Creates a dict structure for the SQL build instructions of a database.

    Returns:
    - dict: Dict of tables and columns with their build instructions.
    """
    tables = execute_query("SELECT name FROM sqlite_master WHERE type='table'")

    schema_dict = {}
    null_column_df = _get_nullable_columns()

    for table in tables:
        table_name = table[0]

        columns = execute_query(f"PRAGMA table_info({table_name})")

        schema_dict[table_name] = {
            'create_table': f'CREATE TABLE {table_name} ({{columns}});',
            'columns': {}
        }

        for _, col_name, data_type, _, col_default, _ in columns:
            col_def = f'{col_name} {data_type.upper()}'

            if col_default:
                col_def += f" DEFAULT {col_default}"

            if f"{table_name}.{col_name}" not in null_column_df['columns'].values:
                col_def += " NOT NULL"

            schema_dict[table_name]['columns'][col_name] = col_def

    return schema_dict


def _get_nullable_columns():
    """
    Return dataframe containing all column names from the DB containing NULL values.
    It contains values from both databases.
    """
    df = pd.DataFrame()

    try:
        df = pd.read_csv('.local/nullable_columns.csv')
    except Exception as e:
        logger.error("Error reading nullable columns from .local/nullable_columns.csv %s", e)

    return df


def _create_build_instruction(build_instruct_dict: dict, tables_columns: dict, level: str) -> str:
    sql_statements = []

    if level == "Full":
        for table, table_info in build_instruct_dict.items():
            create_table_sql = table_info['create_table'].format(
                columns=",\n    ".join(table_info['columns'].values())
            )
            sql_statements.append(create_table_sql)

    elif level == "Tables":
        for table in tables_columns.keys():
            if table in build_instruct_dict:
                table_info = build_instruct_dict[table]
                create_table_sql = table_info['create_table'].format(
                    columns=",\n    ".join(table_info['columns'].values())
                )
                sql_statements.append(create_table_sql)

    elif level == "Columns":
        for table, selected_columns in tables_columns.items():
            if table in build_instruct_dict:
                columns_def = []
                all_cols = build_instruct_dict[table]['columns']

                for col in selected_columns:
                    if col in all_cols:
                        columns_def.append(all_cols[col])

                create_table_sql = build_instruct_dict[table]['create_table'].format(
                    columns=",\n    ".join(columns_def)
                )
                sql_statements.append(create_table_sql)

    return "\n\n".join(sql_statements)
