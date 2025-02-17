from src.core.base_model import NL2SQLModel
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

    def _answer_single_question(self, question):
        answer = self.pipe(
            question, 
            return_full_text=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=1024,
            temperature=0.1,
            top_p=0.8,
            do_sample=True
        )[0]['generated_text']
        return self._prune_generated_query(answer)
    
class DeepSeekQwenModel(NL2SQLModel):
    def __init__(self, connection, benchmark_set, prompt_strategy):
        self.benchmark = benchmark_set
        self.tokenizer = AutoTokenizer.from_pretrained(join(MODELS_DIRECTORY_PATH, "DeepSeekQwen"))
        self.model = AutoModelForCausalLM.from_pretrained(join(MODELS_DIRECTORY_PATH, "DeepSeekQwen"), torch_dtype=torch.bfloat16,
            device_map="auto")
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        self.conn = connection
        self.prompt_strategy = prompt_strategy
        self.results = {}

    def _answer_single_question(self, question):
        answer = self.pipe(
            question, 
            return_full_text=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            temperature=0.6
        )[0]['generated_text']
        return self._prune_generated_query(answer)
    
class DeepSeekLlamaModel(NL2SQLModel):
    def __init__(self, connection, benchmark_set, prompt_strategy):
        self.benchmark = benchmark_set
        self.tokenizer = AutoTokenizer.from_pretrained(join(MODELS_DIRECTORY_PATH, "DeepSeekLlama"))
        self.model = AutoModelForCausalLM.from_pretrained(join(MODELS_DIRECTORY_PATH, "DeepSeekLlama"), torch_dtype=torch.bfloat16,
            device_map="auto")
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        self.conn = connection
        self.prompt_strategy = prompt_strategy
        self.results = {}

    def _answer_single_question(self, question):
        answer = self.pipe(
            question, 
            return_full_text=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            temperature=0.6
        )[0]['generated_text']
        return self._prune_generated_query(answer)
    
class LlamaModel(NL2SQLModel):
    def __init__(self, connection, benchmark_set, prompt_strategy):
        self.benchmark = benchmark_set
        self.tokenizer = AutoTokenizer.from_pretrained(join(MODELS_DIRECTORY_PATH, "Llama"))
        self.model = AutoModelForCausalLM.from_pretrained(join(MODELS_DIRECTORY_PATH, "Llama"), torch_dtype=torch.bfloat16,
            device_map="auto")
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        self.conn = connection
        self.prompt_strategy = prompt_strategy
        self.results = {}

    def _answer_single_question(self, question):
        answer = self.pipe(
            question, 
            return_full_text=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=1024,
        )[0]['generated_text']
        return self._prune_generated_query(answer)