from src.common.logger import get_logger
import sqlite3

from src.database import database

logger = get_logger(__name__)

def get_all_table_names(db_uri: str) -> list[str]:
    conn = sqlite3.connect(db_uri)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    table_names = cursor.fetchall()
    conn.close()
    return [table_name[0] for table_name in table_names]

def get_table_schema_with_samples(
    db_uri: str, table_name: str, sample_limit: int = 0
) -> str:
    conn = sqlite3.connect(db_uri)
    cursor = conn.cursor()

    # Fetch table schema
    cursor.execute(f"PRAGMA table_info(`{table_name}`);")
    columns = cursor.fetchall()
    cursor.execute(f"PRAGMA foreign_key_list(`{table_name}`);")
    foreign_keys = cursor.fetchall()
    cursor.execute(f"PRAGMA index_list(`{table_name}`);")
    primary_key_indices = cursor.fetchall()
    primary_key_columns = []

    for index_info in primary_key_indices:
        index_name = index_info[1]
        cursor.execute(f"PRAGMA index_info(`{index_name}`);")
        index_columns = cursor.fetchall()
        primary_key_columns.extend(column[2] for column in index_columns)

    # Construct CREATE TABLE statement
    schema_str = f"CREATE TABLE `{table_name}` (\n"
    for column in columns:
        column_name = column[1]
        data_type = column[2]
        schema_str += f"  {column_name} {data_type}"
        if column_name in primary_key_columns:
            schema_str += " PRIMARY KEY"
        for foreign_key in foreign_keys:
            if column_name == foreign_key[3]:
                schema_str += f" REFERENCES {foreign_key[2]}({foreign_key[4]})"

        schema_str += ",\n"
    schema_str = schema_str.rstrip(",\n")
    schema_str += "\n);\n"

    
    cursor.execute(f"SELECT * FROM `{table_name}` LIMIT {sample_limit};")
    sample_rows = cursor.fetchall()

    if len(sample_rows) > 0:
        schema_str += f"Sample rows from `{table_name}`:\n"
        for row in sample_rows:
            formatted_row = ", ".join(str(item) for item in row)
            schema_str += f"{formatted_row}\n"

    conn.close()
    return schema_str

def chunk_mschema(mschema: str, tokenizer, with_relations: bool, k: int = 0) -> list[str]:
    """
    Transform the M-Schema into a list of M-Schemas that fits within a given context window.
    Each chunk contains only whole tables from the mschema.

    Args:
    - mschema (str): The mschema of the target database
    - model: Used to get context size and tokenize mschema
    - with_relations (bool): Generate chunks with or without relations
    """
    if with_relations:
        return _chunk_mschema_with_relations(mschema, tokenizer, k)
    return _chunk_mschema_no_relations(mschema, tokenizer, k)


def _chunk_mschema_no_relations(mschema: str, tokenizer, k: int = 0) -> list[str]:
    context_size = _get_context_size(tokenizer)

    if "【Foreign keys】" in mschema:
        mschema = mschema.split("【Foreign keys】")[0]

    mschema_split = mschema.split("# ")
    mschema_header_text = mschema_split[0]
    mschema_tables = ['# ' + table for table in mschema_split[1:]]

    chunks = []
    chunk = set()
    for table in mschema_tables:
        chunk_size = len(tokenizer(' '.join(chunk) + table, return_tensors="pt", truncation=False)["input_ids"][0])

        if (k > 0 and len(chunk) >= k) or chunk_size > context_size / 1.5:
            if chunk:
                chunks.append(mschema_header_text + ' '.join(chunk))
            chunk = set()
            chunk.add(table)
        else:
            chunk.add(table)
    if chunk:
        chunks.append(mschema_header_text + ' '.join(chunk))

    return chunks


def _chunk_mschema_with_relations(mschema: str, tokenizer, k: int) -> list[str]:
    context_size = _get_context_size(tokenizer)

    relations = []
    foreign_key_str = "【Foreign keys】"

    if foreign_key_str in mschema:
        relations = mschema.split(foreign_key_str)[1].split()
        mschema = mschema.split(foreign_key_str)[0]

    mschema_split = mschema.split("# ")
    mschema_header_text = mschema_split[0]
    mschema_tables = ['# ' + table for table in mschema_split[1:]]

    chunks = []

    chunk_tables = set()
    chunk_relations = set()
    for table in mschema_tables:
        chunk_size = len(tokenizer(' '.join(chunk_tables | chunk_relations) + table, return_tensors="pt", truncation=False)["input_ids"][0])

        if (k > 0 and len(chunk_tables) >= k) or chunk_size > context_size // 2:
            if chunk_tables:
                if chunk_relations:
                    chunks.append(mschema_header_text + ' '.join(chunk_tables) + foreign_key_str + '\n' + '\n'.join(chunk_relations))
                else:
                    chunks.append(mschema_header_text + ' '.join(chunk_tables))
            chunk_tables = set()
            chunk_relations = set()
            chunk_tables.add(table)
            _find_relations(table, chunk_tables, chunk_relations, mschema_tables, relations, context_size, tokenizer)
        else:
            chunk_tables.add(table)
            _find_relations(table, chunk_tables, chunk_relations, mschema_tables, relations, context_size, tokenizer)
    if chunk_tables:
        if chunk_relations:
            chunks.append(mschema_header_text + ' '.join(chunk_tables) + foreign_key_str + '\n' + '\n'.join(chunk_relations))
        else:
            chunks.append(mschema_header_text + ' '.join(chunk_tables))
    return chunks


def _find_relations(table: str, chunk_tables: set[str], chunk_relations: set[str], mschema_tables: list[str], relations: list[str], context_size: int, tokenizer) -> None:
    """
    Identifies all relevant for foreign key relations for a given table in the M-Schema
    Uses the relations to add the relations and their tables to the schema chunk.
    """
    table_name = table.split('# Table:')[1].split('[')[0].strip()

    table_relations = {relation for relation in relations if table_name == relation.split('.')[0]}

    # Identify tables from the relations and add to the M-Schema chunk
    for relation in table_relations:
        table_relation_name = relation.split('=')[1].split('.')[0]
        for mschema_table in mschema_tables:
            if "# Table: " + table_relation_name in mschema_table:
                # Ensure that max_mschema_size is not exceded due to adding all relation tables.
                chunk_size = len(tokenizer(' '.join(chunk_tables | chunk_relations) + mschema_table, return_tensors="pt", truncation=False)["input_ids"][0])
                if chunk_size > context_size // 1.5:
                    break
                chunk_tables.add(mschema_table)

    chunk_relations.update(table_relations)


def _get_context_size(tokenizer) -> int:
    context_size = getattr(tokenizer, "model_max_length", None)
    if context_size is None:
        logger.error("Could not get model context size.")
        raise ValueError("Model context size (max_position_embeddings) is not set.")

    return context_size


def chunk_dts_ddl(ddl_schema: str, tokenizer, k: int = 0) -> list[str]:
    """
    Chunks DDL schema, as defined by dts_sql implenentation, into smaller chunks.
    Args:
        - ddl_schema (str)
        - toknizer: Model tokenizer
        - k (int): The amount of tables in each chunk
    """
    context_size = _get_context_size(tokenizer)
    split_schema = ddl_schema.split(";")
    split_schema.pop() # Remove last empty entry due to splitting on ';'
    tables = [table + ';' for table in split_schema]

    chunks = []
    chunk = set()
    for table in tables:
        chunk_size = len(tokenizer(' '.join(chunk) + table, return_tensors="pt", truncation=False)["input_ids"][0])
        if (k > 0 and len(chunk) >= k) or chunk_size > context_size / 1.5:
            if chunk:
                chunks.append(' '.join(chunk))
            chunk = set()
            chunk.add(table)
        else:
            chunk.add(table)
    if chunk:
        chunks.append(' '.join(chunk))

    return chunks

def chunk_dts_ddl_relations(ddl_schema: str, tokenizer, k: int = 0) -> list[str]:
    """
    Chunks DDL schema, as defined by dts_sql implenentation, into smaller chunks with reltions.
    Args:
        - ddl_schema (str)
        - toknizer: Model tokenizer
        - k (int): The amount of tables in each chunk
    """
    # context_size = _get_context_size(tokenizer)
    context_size = 10000

    split_schema = ddl_schema.split(";")
    split_schema.pop() # Remove last empty entry due to splitting on ';'
    tables = [table + ';' for table in split_schema]

    chunks = []
    chunk = set()
    for table in tables:
        # chunk_size = len(tokenizer(' '.join(chunk) + table, return_tensors="pt", truncation=False)["input_ids"][0])
        chunk_size = len(" ".join(chunk)) + len(table)

        if (k > 0 and len(chunk) >= k) or chunk_size > context_size // 2:
            if chunk:
                chunks.append(' '.join(chunk))
            chunk = set()
            chunk.add(table)
            find_relations_tables_dts(table, chunk, tables, context_size, tokenizer)
        else:
            chunk.add(table)
            find_relations_tables_dts(table, chunk, tables, context_size, tokenizer)
    if chunk:
        chunks.append(' '.join(chunk))

    return chunks

def find_relations_tables_dts(table: str, chunk: set[str], tables: list[str], context_size: int, tokenizer) -> None:
    table_reference_split = table.split('REFERENCES')[1:]
    table_references = {reference.split('(')[0].strip() for reference in table_reference_split}

    for table in tables:
        for table_reference_name in table_references:
            if f"TABLE `{table_reference_name}`" in table:
                # chunk_size = len(tokenizer(' '.join(chunk) + table, return_tensors="pt", truncation=False)["input_ids"][0])
                chunk_size = len(' '.join(chunk)) + len(table)
                if chunk_size < context_size // 1.5:
                    chunk.add(table)
                else:
                    return


if __name__ == "__main__":
    db_uri = ".local/train/train_databases/works_cycles/works_cycles.sqlite"
    table_names = get_all_table_names(db_uri)
    database_schema = ""
    for table_name in table_names:
        database_schema = database_schema + get_table_schema_with_samples(db_uri, table_name, 0) + '\n'
    print(len(database_schema))

    chunks = chunk_dts_ddl_relations(database_schema, 10000)
    print(len(chunks))

    print(len(' '.join(chunks)))
