# import torch
from sqlalchemy import create_engine
from mschema.schema_engine import SchemaEngine


def chunk_mschema_no_relations(mschema: str, model) -> list[str]:
    """
    Transform the M-Schema into a list of M-Schemas that fits within a given context window.
    Each chunk contains only whole tables from the mschema.

    Args:
    - model: Used to get context size and tokenize mschema.
    - context_size: The size of a models context window
    """

    context_size = 10000
    # context_size = getattr(model.model.config, "max_position_embeddings", None)
    # if context_size is None:
    #     raise ValueError("Model context size (max_position_embeddings) is not set.")

    max_mschema_size = context_size // 1.5

    mschema_split = mschema.split("#")
    mschema_header_text = mschema_split[0]
    mschema_tables = ['#' + table for table in mschema_split[1:]]

    if "【Foreign keys】" in mschema_tables[-1]:
        mschema_tables[-1] = mschema_tables[-1].split("【Foreign keys】")[0]

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


def generate_mschema():
    db_engine = create_engine('sqlite:///.local/train/train_databases/works_cycles/works_cycles.sqlite')
    with open(".local/mschema_chunking_test.txt", "w") as file:
        file.write(SchemaEngine(engine=db_engine, db_name="works_cycles").mschema.to_mschema())


if __name__ == "__main__":
    mschema: str

    generate_mschema()

    with open(".local/mschema_chunking_test.txt", "r") as file:
        mschema = file.read()

    chunks = chunk_mschema_no_relations(mschema, 1)
    for chunk in chunks:
        print(chunk)
    print(len(chunks))
