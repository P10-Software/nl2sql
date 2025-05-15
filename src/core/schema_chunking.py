# import torch
from sqlalchemy import create_engine
from mschema.schema_engine import SchemaEngine


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

    mschema_split = mschema.split("#")
    mschema_header_text = mschema_split[0]
    mschema_tables = ['#' + table for table in mschema_split[1:]]

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

    if "【Foreign keys】" in mschema:
        relation = mschema.split("【Foreign keys】")[1].split()
        mschema = mschema.split("【Foreign keys】")[0]

    mschema_split = mschema.split("#")
    mschema_header_text = mschema_split[0]
    mschema_tables = ['#' + table for table in mschema_split[1:]]

    chunks = []
    chunk_relations = set()
    chunk = set()
    new_chunk_size = 0
    for table in mschema_tables:
        # table_size = len(model.tokenizer(table, return_tensors="pt", truncation=False)["input_ids"][0])
        table_size = len(table)
        new_chunk_size = new_chunk_size + table_size
        # find_relations(table, chunk_relations, chunk, relations)
        if new_chunk_size > max_mschema_size:
            if chunk:
                chunks.append(mschema_header_text + ' '.join(chunk))
            chunk = set()
            chunk.add(table)
            new_chunk_size = table_size
        else:
            chunk.add(table)

    if chunk:
        chunks.append(mschema_header_text + ' '.join(chunk))

    return chunks


def find_relations(table, chunk_relations, chunk, relations):
    chunk_relations = []
    table_name = table.split('# Table:')[1].split('[')[0].strip()
    print(table_name)


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
