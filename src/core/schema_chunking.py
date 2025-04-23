import torch
from src.core.model_implementations import LlamaModel, DeepSeekLlamaModel, DeepSeekQwenModel, XiYanSQLModel
from src.core.prompt_strategies import Llama3PromptStrategy, DeepSeekPromptStrategy, XiYanSQLPromptStrategy
from src.core.base_model import NL2SQLModel
#

def chunk_mschema(mschema: str, model) -> list[str]:
    """
    Transform the M-Schema into a list of M-Schemas that fits within a given context window.

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


# def test_token(mschema):
#     dataset = [{"question": "test question", "golden_query": "test gold"}]
#     prompt_strategy = XiYanSQLPromptStrategy("sqlite")
#     model = XiYanSQLModel(dataset, prompt_strategy, mschema=True)
#
#     #mschema = mschema.split("#")
#
#     prompt = model.prompt_strategy.get_prompt(mschema, "This is a question template")
#
#     print(prompt)
#     print("Characters in prompt: ", len(prompt))
#     tokens = model.tokenizer(prompt, return_tensors="pt", truncation=False)["input_ids"][0]
#     print("Tokens in prompt: ", len(tokens))
#     context_length = getattr(model.model.config, "max_position_embeddings", None)
#     print("Model context length: ", context_length)


if __name__ == "__main__":
    with open(".local/mschema_trial_metadata_abbreviated.txt", "r") as file:
        mschema = file.read()

    dataset = [{"question": "test question", "golden_query": "test gold"}]
    prompt_strategy = XiYanSQLPromptStrategy("sqlite")
    model = XiYanSQLModel(dataset, prompt_strategy, mschema=True)

    chunks = chunk_mschema(mschema, model)

    print("Chunk sizes")
    print([len(x) for x in chunks])
