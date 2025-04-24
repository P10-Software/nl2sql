import torch


def chunk_mschema(mschema: str, model) -> list[str]:
    """
    Transform the M-Schema into a list of M-Schemas that fits within a given context window.
    Each chunk contains only whole tables from the mschema.

    Args:
    - model: Used to get context size and tokenize mschema.
    - context_size: The size of a models context window
    """

    context_size = getattr(model.model.config, "max_position_embeddings", None)
    if context_size is None:
        raise ValueError("Model context size (max_position_embeddings) is not set.")

    max_mschema_size = context_size // 1.5

    mschema_split = mschema.split("#")
    mschema_header_text = mschema_split[0]
    mschema_tables = ['#' + table for table in mschema_split[1:]]

    chunks = []
    chunk = ""
    new_chunk_size = 0
    for table in mschema_tables:
        table_size = len(model.tokenizer(table, return_tensors="pt", truncation=False)["input_ids"][0])
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
