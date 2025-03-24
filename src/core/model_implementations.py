from os.path import join
import re
import torch
from src.core.base_model import NL2SQLModel
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

MODELS_DIRECTORY_PATH = "models/"

class XiYanSQLModel(NL2SQLModel):
    def __init__(self, benchmark_set, prompt_strategy, mschema: bool=False):
        super().__init__(benchmark_set, prompt_strategy, mschema)
        self.tokenizer = AutoTokenizer.from_pretrained(join(MODELS_DIRECTORY_PATH, "XiYanSQL"))
        self.model = AutoModelForCausalLM.from_pretrained(join(MODELS_DIRECTORY_PATH, "XiYanSQL"), torch_dtype=torch.bfloat16, device_map="auto")
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def _answer_single_question(self, question, schema):
        prompt = self.prompt_strategy.get_prompt(schema, question)
        return self._prune_generated_query(self.pipe(
            prompt,
            return_full_text=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=1024,
            temperature=0.1,
            top_p=0.8,
            do_sample=True
        )[0]['generated_text'])

class DeepSeekQwenModel(NL2SQLModel):
    def __init__(self, benchmark_set, prompt_strategy, mschema: bool=False):
        super().__init__(benchmark_set, prompt_strategy, mschema)
        self.tokenizer = AutoTokenizer.from_pretrained(join(MODELS_DIRECTORY_PATH, "DeepSeekQwen"))
        self.model = AutoModelForCausalLM.from_pretrained(join(MODELS_DIRECTORY_PATH, "DeepSeekQwen"), device_map="auto", torch_dtype=torch.bfloat16)
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def _answer_single_question(self, question, schema):
        prompt = self.prompt_strategy.get_prompt(schema, question)
        return self._prune_generated_query(self.pipe(
            prompt,
            return_full_text=False,
            max_new_tokens=1024,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            temperature=0.6
        )[0]['generated_text'])

    def _prune_generated_query(self, query):
        query = super()._prune_generated_query(query)
        query = re.sub(r".*SELECT", "SELECT", query, flags=re.IGNORECASE)
        return re.sub(r'([a-z])([A-Z])', r'\1 \2', query)

class DeepSeekLlamaModel(NL2SQLModel):
    def __init__(self, benchmark_set, prompt_strategy, mschema: bool=False):
        super().__init__(benchmark_set, prompt_strategy, mschema)
        self.tokenizer = AutoTokenizer.from_pretrained(join(MODELS_DIRECTORY_PATH, "DeepSeekLlama"))
        self.model = AutoModelForCausalLM.from_pretrained(join(MODELS_DIRECTORY_PATH, "DeepSeekLlama"), device_map="auto", torch_dtype=torch.bfloat16)
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def _answer_single_question(self, question, schema):
        prompt = self.prompt_strategy.get_prompt(schema, question)
        return self._prune_generated_query(self.pipe(
            prompt,
            return_full_text=False,
            max_new_tokens=1024,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            temperature=0.6
        )[0]['generated_text'])

class LlamaModel(NL2SQLModel):
    def __init__(self, benchmark_set, prompt_strategy, mschema: bool=False):
        super().__init__(benchmark_set, prompt_strategy, mschema)
        self.tokenizer = AutoTokenizer.from_pretrained(join(MODELS_DIRECTORY_PATH, "Llama"))
        self.model = AutoModelForCausalLM.from_pretrained(join(MODELS_DIRECTORY_PATH, "Llama"), device_map="auto", torch_dtype=torch.bfloat16)
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def _answer_single_question(self, question, schema):
        prompt = self.prompt_strategy.get_prompt(schema, question)
        return self._prune_generated_query(self.pipe(
            prompt,
            return_full_text=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=1024,
        )[0]['generated_text'])

    def _prune_generated_query(self, query: str):
        if ';' in query:
            return super()._prune_generated_query(query)
        else:
            return super()._prune_generated_query(query) + ";"

class ModelWithSQLCoderAbstentionModule(NL2SQLModel):
    def __init__(self, benchmark_set, prompt_strategy,  sql_generation_model: NL2SQLModel, pre_sql_abstention: bool, post_sql_abstention: bool, mschema: bool = False):
        super().__init__(benchmark_set, prompt_strategy, mschema)
        self.abstention_tokenizer = AutoTokenizer.from_pretrained(join(MODELS_DIRECTORY_PATH, "SQLCoder"))
        self.abstention_model = AutoModelForCausalLM.from_pretrained(join(MODELS_DIRECTORY_PATH, "SQLCoder"), device_map="auto", torch_dtype=torch.float16)
        self.abstention_pipe = pipeline("text-generation", model=self.abstention_model, tokenizer=self.abstention_tokenizer)
        self.pre_sql_abstention = pre_sql_abstention
        self.post_sql_abstention = post_sql_abstention
        self.sql_generation_model = sql_generation_model

    def _answer_single_question(self, question, schema):
        if self.pre_sql_abstention:
            pre_abstention_prompt = self.prompt_strategy.get_prompt(schema, question)
            pre_abstention_answer = self.abstention_pipe(
                pre_abstention_prompt,
                return_full_text=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=1024,
                do_sample=False,
                num_beams=4
            )

            if "I do not know" in pre_abstention_answer:
                return None

        sql_answer = self.sql_generation_model._answer_single_question(question, schema)

        if self.post_sql_abstention:
            post_abstention_prompt = self.prompt_strategy.get_prompt(schema, question, sql_answer)
            post_abstention_answer = self.abstention_pipe(
                post_abstention_prompt,
                return_full_text=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=1024,
                do_sample=False,
                num_beams=4
            )

            if "incorrect" in post_abstention_answer:
                return None

        return sql_answer
