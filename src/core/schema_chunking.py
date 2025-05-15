# import torch
from sqlalchemy import create_engine
from mschema.schema_engine import SchemaEngine
from src.common.logger import get_logger

logger = get_logger(__name__)


def chunk_mschema_no_relations(mschema: str, model) -> list[str]:
    """
    Transform the M-Schema into a list of M-Schemas that fits within a given context window.
    Each chunk contains only whole tables from the mschema.
    Does not contain foreign key relations.

    Args:
    - model: Used to get context size and tokenize mschema
    - context_size: The size of a models context window
    """

    context_size = 10000
    # context_size = getattr(model.model.config, "max_position_embeddings", None)
    # if context_size is None:
    #     raise ValueError("Model context size (max_position_embeddings) is not set.")

    max_mschema_size = context_size // 1.5

    if "【Foreign keys】" in mschema:
        mschema = mschema.split("【Foreign keys】")[0]

    mschema_split = mschema.split("# ")
    mschema_header_text = mschema_split[0]
    mschema_tables = ['# ' + table for table in mschema_split[1:]]

    chunks = []
    chunk = ""
    new_chunk_size = 0
    for table in mschema_tables:
        # table_size = len(model.tokenizer(table, return_tensors="pt", truncation=False)["input_ids"][0])
        table_size = len(table)
        new_chunk_size = new_chunk_size + table_size
        if new_chunk_size > max_mschema_size:
            if chunk:
                chunks.append(mschema_header_text + chunk)
            chunk = table
            new_chunk_size = table_size
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
    - model: Used to get context size and tokenize mschema
    - context_size: The size of a models context window
    """

    context_size = 10000
    # context_size = getattr(model.model.config, "max_position_embeddings", None)
    # if context_size is None:
    #     raise ValueError("Model context size (max_position_embeddings) is not set.")

    max_mschema_size = context_size // 1.5

    relations = []
    foreign_key_str = "【Foreign keys】"

    if "【Foreign keys】" in mschema:
        relations = mschema.split("【Foreign keys】")[1].split()
        mschema = mschema.split("【Foreign keys】")[0]

    mschema_split = mschema.split("# ")
    mschema_header_text = mschema_split[0]
    mschema_tables = ['# ' + table for table in mschema_split[1:]]

    chunks = []
    chunk_tables = set()
    chunk_relations = set()
    new_chunk_size = 0
    for table in mschema_tables:
        # table_size = len(model.tokenizer(table, return_tensors="pt", truncation=False)["input_ids"][0])
        # TODO Consider when to calculate chunk size as it might grow significantly.
        table_size = len(table)
        new_chunk_size = new_chunk_size + table_size
        find_relations(table, chunk_tables, chunk_relations, mschema_tables, relations)
        chunk_tables.add(table)
        if new_chunk_size > max_mschema_size:
            if chunk_tables:
                chunks.append(mschema_header_text + ' '.join(chunk_tables) + foreign_key_str + '\n' + '\n'.join(chunk_relations))
            chunk_tables = set()
            chunk_relations = set()
            chunk_tables.add(table)
            new_chunk_size = table_size
        else:
            chunk_tables.add(table)

    if chunk_tables:
        chunks.append(mschema_header_text + ' '.join(chunk_tables) + foreign_key_str + '\n' + '\n'.join(chunk_relations))

    return chunks


def find_relations(table, chunk_tables, chunk_relations, mschema_tables, relations):
    table_name = table.split('# Table:')[1].split('[')[0].strip()

    table_relations = {relation for relation in relations if table_name == relation.split('.')[0]}

    # Identify tables from the relations and add to the M-Schema chunk
    for relation in table_relations:
        table_relation_name = relation.split('=')[1].split('.')[0]
        for table in mschema_tables:
            if "# Table: " + table_relation_name in table:
                chunk_tables.add(table)

    chunk_relations.update(table_relations)


# def generate_mschema():
#     db_engine = create_engine('sqlite:///.local/train/train_databases/works_cycles/works_cycles.sqlite')
#     with open(".local/mschema_chunking_test.txt", "w") as file:
#         file.write(SchemaEngine(engine=db_engine, db_name="works_cycles").mschema.to_mschema())


if __name__ == "__main__":
    with open(".local/mschema_chunking_test.txt", "r") as file:
        mschema = file.read()

    chunks = chunk_mschema(mschema, 1)
    for chunk in chunks:
        print(chunk)
    print(len(chunks))
