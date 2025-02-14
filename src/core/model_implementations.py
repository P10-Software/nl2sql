from src.core.base_model import NL2SQLModel
from src.core.prompt_strategies import XiYanSQLPromptStrategy, Llam3PromptStrategy, DeepSeekPromptStrategy
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from os.path import join
import torch

MODELS_DIRECTORY_PATH = "/work/P10_FILE_STORAGE/models/"

class XiYanSQLModel(NL2SQLModel):
    def __init__(self, connection, benchmark_set, prompt_strategy):
        self.benchmark = benchmark_set
        self.tokenizer = AutoTokenizer.from_pretrained(join(MODELS_DIRECTORY_PATH, "XiYanSQL"))
        self.model = AutoModelForCausalLM.from_pretrained(join(MODELS_DIRECTORY_PATH, "XiYanSQL"), torch_dtype=torch.bfloat16,
            device_map="auto")
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        self.conn = connection
        self.prompt_strategy = prompt_strategy
        self.results = {}