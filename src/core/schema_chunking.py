import torch
from src.core.model_implementations import LlamaModel, DeepSeekLlamaModel, DeepSeekQwenModel, XiYanSQLModel
from src.core.prompt_strategies import Llama3PromptStrategy, DeepSeekPromptStrategy, XiYanSQLPromptStrategy
from src.core.base_model import NL2SQLModel


def chunk_mschema(mschema: str, context_size) -> list[str]:
    """
    Transform the M-Schema into a list of M-Schemas that fits within a given context window

    Args:
    - mschema (str): DB schema in M-Schema format.
    - context_size: The size of a models context window
    """
    
    test = mschema.split("#")
    print(test[92])
    # [print(x) for x in test]

    return [mschema]

if __name__ == "__main__":
    mschema = ""
    with open(".local/mschema_abbreviated.txt", "r") as file:
        mschema = file.read()

        dataset = {"question": "test question", "golden_query": "test gold"}
        prompt_strategy = XiYanSQLPromptStrategy("sqlite")
        model = XiYanSQLModel(dataset, prompt_strategy, mschema=True)

    print("Chunking")

    chunk_mschema(mschema, 1000)
