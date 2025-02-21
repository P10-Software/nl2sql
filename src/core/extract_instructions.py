from typing import Literal
import re
import pandas as pd
from sql_metadata import Parser
from src.common.logger import get_logger
from src.database.setup_database import get_conn


SchemaKind = Literal['full', 'tables', 'columns']
logger = get_logger(__name__)


def get_query_build_instruct(kind: SchemaKind, query: str, natural_names: bool) -> str:
    """
    Find the build instructions of the database based on a query.

    Args:
    - kind (SchemaKind): One of 'full', 'tables', 'columns' specifying how restricted the schema should be.
    - query: SQL query in string format
    - natural_names: Boolean values, True for natural_names and False for abbreviated

    Returns:
    - SQL build instructions (sql): SQL instructions specifying how to build the DB.
    """
    conn = get_conn()
    selected_tables_columns = _extract_column_table(query)

    if natural_names:
        selected_tables_columns = _transform_natural_query(selected_tables_columns)

    schema_tree = _create_build_instruction_tree(conn)

    return _create_build_instruction(schema_tree, selected_tables_columns, kind)


def _extract_column_table(query: str) -> dict[str, list[str]]:
    parser = Parser(sanitise_query(query))
    tables = parser.tables
    columns = _parse_column(parser)

    column_table_mapping = {}

    if len(tables) == 1:
        single_table = tables[0]
        for col in columns:
            if '.' in col:
                table_name, column_name = col.split('.')
                column_table_mapping.setdefault(
                    table_name, []).append(column_name)
            else:
                column_table_mapping.setdefault(single_table, []).append(col)
    else:
        for col in columns:
            if '.' in col:
                table_name, column_name = col.split('.')
                column_table_mapping.setdefault(
                    table_name, []).append(column_name)
            else:
                logger.error("ERROR extracting columns, found ambiguity.")
                raise RuntimeError(f"Ambiguity found in query {query}, quitting.")

    return column_table_mapping


def _transform_natural_query(selected_tables_columns: dict[str, list[str]]) -> dict[str, list[str]]:
    """ Transform tables and column names in query to be more natural. """
    table_names = pd.read_csv(".local/table_names_normalised.csv", header=None, names=["old_name", "new_name"])
    column_names = pd.read_csv(".local/column_names_normalised.csv", header=None, names=["old_name", "new_name"])

    table_mapping = dict(zip(table_names['old_name'], table_names['new_name']))
    column_mapping = dict(zip(column_names['old_name'], column_names['new_name']))
    
    updated_dict = {}

    for key, values in selected_tables_columns.items():
        # Replace key if found, otherwise keep the original
        new_table = table_mapping.get(key, key)

        new_values = [column_mapping.get(val, val) for val in values]

        updated_dict[new_table] = new_values

    return updated_dict


def sanitise_query(query: str):
    return re.sub(r"(LIKE\s*)'[^']*'", r"\1''", query, flags=re.IGNORECASE)


def _parse_column(parser: Parser):
    columns = parser.columns
    if all('.' in col for col in columns):
        return columns

    columns = []

    for token in parser.tokens:
        if token.is_keyword and token.normalized == 'SELECT':
            next_token = token.next_token
            column_names = []
            while next_token is not None:
                if next_token.value not in ['.', ',']:
                    column_names.append(next_token.value)
                next_token = next_token.next_token
                if next_token.normalized == 'FROM':
                    columns.extend(
                        [(next_token.next_token.value + '.' + s if '.' not in s else s) for s in column_names])
                    break

    return columns


def _create_build_instruction_tree(connection_string) -> dict:
    """
    Creates a dict structure for the SQL build instructions of a database.
    Parameters:
    - conn: PSQL connection string.
    Returns:
    - dict: Dict of tables and columns with their build instructions.
    """
    conn = connection_string
    cursor = conn.cursor()

    cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_type = 'BASE TABLE'")
    tables = cursor.fetchall()

    schema_dict = {}

    for table in tables:
        table_name = table[0]
        cursor.execute(f"SELECT column_name, data_type, character_maximum_length, is_nullable, column_default FROM information_schema.columns WHERE table_name = '{table_name}'")
        columns = cursor.fetchall()

        schema_dict[table_name] = {
            'create_table': f'CREATE TABLE {table_name} ({{columns}});',
            'columns': {}
        }

        for col_name, data_type, char_length, is_nullable, col_default in columns:
            col_def = f'{col_name} {data_type.upper()}'

            if char_length and data_type in ('character varying', 'varchar'):
                col_def += f"({char_length})"

            if col_default:
                col_def += f" DEFAULT {col_default}"

            if is_nullable == 'NO':
                col_def += " NOT NULL"

            schema_dict[table_name]['columns'][col_name] = col_def
    cursor.close()
    conn.close()

    return schema_dict


def _create_build_instruction(build_instruct_dict: dict, tables_columns: dict, level: str) -> str:
    sql_statements = []

    if level == "full":
        for table, table_info in build_instruct_dict.items():
            create_table_sql = table_info['create_table'].format(
                columns=",\n    ".join(table_info['columns'].values())
            )
            sql_statements.append(create_table_sql)

    elif level == "tables":
        for table in tables_columns.keys():
            if table in build_instruct_dict:
                table_info = build_instruct_dict[table]
                create_table_sql = table_info['create_table'].format(
                    columns=",\n    ".join(table_info['columns'].values())
                )
                sql_statements.append(create_table_sql)

    elif level == "columns":
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
