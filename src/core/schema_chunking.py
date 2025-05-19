import numpy
from src.common.logger import get_logger

logger = get_logger(__name__)

def chunk_mschema(mschema: str, model, with_relations: bool) -> list[str]:
    """
    Transform the M-Schema into a list of M-Schemas that fits within a given context window.
    Each chunk contains only whole tables from the mschema.

    Args:
    - mschema (str): The mschema of the target database
    - model: Used to get context size and tokenize mschema
    - with_relations (bool): Generate chunks with or without relations
    """
    if with_relations:
        return _chunk_mschema_with_relations(mschema, model)
    return _chunk_mschema_no_relations(mschema, model)


def _chunk_mschema_no_relations(mschema: str, model) -> list[str]:
    context_size = _get_model_context_size(model.tokenizer)

    if "【Foreign keys】" in mschema:
        mschema = mschema.split("【Foreign keys】")[0]

    mschema_split = mschema.split("# ")
    mschema_header_text = mschema_split[0]
    mschema_tables = ['# ' + table for table in mschema_split[1:]]

    chunks = []
    chunk = ""
    for table in mschema_tables:
        chunk_size = len(model.tokenizer(chunk + table, return_tensors="pt", truncation=False)["input_ids"][0])

        if chunk_size > context_size / 1.5:
            if chunk:
                chunks.append(mschema_header_text + chunk)
            chunk = table
        else:
            chunk = chunk + table
    if chunk:
        chunks.append(mschema_header_text + chunk)

    return chunks

def mschema_to_k_chunks(mschema: str, tokenizer, k: int) -> list[str]:
    max_chunk_size = _get_model_context_size(tokenizer) // 1.5

    if "【Foreign keys】" in mschema:
        mschema = mschema.split("【Foreign keys】")[0]

    mschema_split = mschema.split("#")
    mschema_header_text = mschema_split[0]
    mschema_tables = ['#' + table for table in mschema_split[1:]]
    amount_of_tables = len(mschema_tables)
    if k > amount_of_tables:
        k = amount_of_tables

    chunks = [mschema_header_text + "".join(x.tolist()) for x in numpy.array_split(mschema_tables, k)]
    for chunk in chunks:
        if len(tokenizer(chunk, return_tensors="pt", truncation=False)["input_ids"][0]) > max_chunk_size:
            raise Exception("Chunk does not fit into model")

    return chunks

def _chunk_mschema_with_relations(mschema: str, model) -> list[str]:
    context_size = _get_model_context_size(model.tokenizer)

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
        chunk_size = len(model.tokenizer(' '.join(chunk_tables | chunk_relations) + table, return_tensors="pt", truncation=False)["input_ids"][0])

        if chunk_size > context_size // 2:
            if chunk_tables:
                chunks.append(mschema_header_text + ' '.join(chunk_tables) + foreign_key_str + '\n' + '\n'.join(chunk_relations))
            chunk_tables = set()
            chunk_relations = set()
            chunk_tables.add(table)
            _find_relations(table, chunk_tables, chunk_relations, mschema_tables, relations, context_size, model)
        else:
            chunk_tables.add(table)
            _find_relations(table, chunk_tables, chunk_relations, mschema_tables, relations, context_size, model)
    if chunk_tables:
        chunks.append(mschema_header_text + ' '.join(chunk_tables) + foreign_key_str + '\n' + '\n'.join(chunk_relations))

    return chunks


def _find_relations(table: str, chunk_tables: set[str], chunk_relations: set[str], mschema_tables: list[str], relations: list[str], context_size: int, model) -> None:
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
                chunk_size = len(model.tokenizer(' '.join(chunk_tables | chunk_relations) + mschema_table, return_tensors="pt", truncation=False)["input_ids"][0])
                if chunk_size > context_size // 1.5:
                    break
                chunk_tables.add(mschema_table)

    chunk_relations.update(table_relations)


def _get_model_context_size(tokenizer) -> int:
    context_size = getattr(tokenizer, "model_max_length", None)
    if context_size is None:
        logger.error("Could not get model context size.")
        raise ValueError("Model context size (max_position_embeddings) is not set.")

    return context_size
