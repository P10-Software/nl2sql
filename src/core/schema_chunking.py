# import torch
from src.common.logger import get_logger

logger = get_logger(__name__)


def chunk_mschema_no_relations(mschema: str, model) -> list[str]:
    """
    Transform the M-Schema into a list of M-Schemas that fits within a given context window.
    Each chunk contains only whole tables from the mschema.
    Does not contain foreign key relations.

    Args:
    - mschema (str): The mschema of the target database
    - model: Used to get context size and tokenize mschema
    """

    # context_size = 10000
    context_size = getattr(model.model.config, "max_position_embeddings", None)
    if context_size is None:
        raise ValueError("Model context size (max_position_embeddings) is not set.")

    max_mschema_size = context_size // 1.5

    if "【Foreign keys】" in mschema:
        mschema = mschema.split("【Foreign keys】")[0]

    mschema_split = mschema.split("# ")
    mschema_header_text = mschema_split[0]
    mschema_tables = ['# ' + table for table in mschema_split[1:]]

    chunks = []
    chunk = ""
    chunk_size = 0
    for table in mschema_tables:
        table_size = len(model.tokenizer(table, return_tensors="pt", truncation=False)["input_ids"][0])
        # table_size = len(table)
        chunk_size = chunk_size + table_size
        if chunk_size > max_mschema_size:
            if chunk:
                chunks.append(mschema_header_text + chunk)
            chunk = table
            chunk_size = table_size
        else:
            chunk = chunk + table

    if chunk:
        chunks.append(mschema_header_text + chunk)

    return chunks


def chunk_mschema(mschema: str, model) -> list[str]:
    """
    Transform the M-Schema into a list of M-Schemas that fits within a given context window.
    Each chunk contains only whole tables from the mschema.

    Args:
    - mschema (str): The mschema of the target database
    - model: Used to get context size and tokenize mschema
    """

    context_size = 10000
    # context_size = getattr(model.model.config, "max_position_embeddings", None)
    # if context_size is None:
    #     raise ValueError("Model context size (max_position_embeddings) is not set.")

    # max_mschema_size = context_size // 2

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
    chunk_size = 0
    for table in mschema_tables:
        # table_size = len(model.tokenizer(table, return_tensors="pt", truncation=False)["input_ids"][0])

        chunk_size = len(' '.join(chunk_tables | chunk_relations)) + len(table)
        if chunk_size > context_size // 2:
            if chunk_tables:
                chunks.append(mschema_header_text + ' '.join(chunk_tables) + foreign_key_str + '\n' + '\n'.join(chunk_relations))
            chunk_tables = set()
            chunk_relations = set()
            chunk_size = 0
            chunk_tables.add(table)
            _find_relations(table, chunk_tables, chunk_relations, mschema_tables, relations, context_size)
        else:
            chunk_tables.add(table)
            _find_relations(table, chunk_tables, chunk_relations, mschema_tables, relations, context_size)

    if chunk_tables:
        chunks.append(mschema_header_text + ' '.join(chunk_tables) + foreign_key_str + '\n' + '\n'.join(chunk_relations))

    return chunks


def _find_relations(table: str, chunk_tables: set[str], chunk_relations: set[str], mschema_tables: list[str], relations: list[str], context_size: int) -> None:
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
                if len(' '.join(chunk_tables | chunk_relations)) + len(mschema_table) > context_size // 1.5:
                    break
                chunk_tables.add(mschema_table)

    chunk_relations.update(table_relations)


if __name__ == "__main__":
    with open(".local/mschema_chunking_test.txt", "r") as file:
        mschema = file.read()

    chunks = chunk_mschema(mschema, 1)
    for chunk in chunks:
        print(len(chunk))
        # print(chunk)
    print(len(chunks))
