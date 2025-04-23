# import torch
# from src.core.model_implementations import LlamaModel, DeepSeekLlamaModel, DeepSeekQwenModel, XiYanSQLModel
# from src.core.prompt_strategies import Llama3PromptStrategy, DeepSeekPromptStrategy, XiYanSQLPromptStrategy
# from src.core.base_model import NL2SQLModel
#

def chunk_mschema(mschema: str, context_size):
    """
    Transform the M-Schema into a list of M-Schemas that fits within a given context window

    Args:
    - mschema (str): DB schema in M-Schema format.
    - context_size: The size of a models context window
    """

    max_mschema_size = context_size // 2

    mschema_split = mschema.split("#")
    mschema_header_text = mschema_split[0]

    print("header size: ", len(mschema_header_text))

    tables = ['#' + table for table in mschema_split[1:]]
    tables_tokens_size = [len(table) for table in tables]

    # print(tables[1])
    # print(mschema)

    chunks = []
    chunk = ""
    new_chunk_size = 0
    for i, table_size in enumerate(tables_tokens_size):
        new_chunk_size = new_chunk_size + table_size
        if new_chunk_size > max_mschema_size:
            print(new_chunk_size)
            chunks.append(chunk)
            # break
            chunk = tables[i]
            new_chunk_size = 0
        else:
            chunk = chunk + tables[i]

    chunks.append(chunk)

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
    mschema = ""
    with open(".local/mschema_trial_metadata_abbreviated.txt", "r") as file:
        mschema = file.read()


    chunks = chunk_mschema(mschema, 10000)

    print("Mschema length: ", len(mschema))
    print("Chunked length: ", len(' '.join(chunks)))
    # with open("regular_mschema.txt", "w") as file:
    #     file.write(mschema)
    # with open("chunked_mschema.txt", "w") as file:
    #     file.write(' '.join(chunks))

    print("Chunk sizes")
    print([len(x) for x in chunks])
