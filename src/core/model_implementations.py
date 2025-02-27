from src.core.base_model import NL2SQLModel
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from os.path import join
import torch
import re
import numpy as np

MODELS_DIRECTORY_PATH = "models/"

class XiYanSQLModel(NL2SQLModel):
    def __init__(self, connection, benchmark_set, prompt_strategy):
        super().__init__(connection, benchmark_set, prompt_strategy)
        self.tokenizer = AutoTokenizer.from_pretrained(join(MODELS_DIRECTORY_PATH, "XiYanSQL"))
        self.model = AutoModelForCausalLM.from_pretrained(join(MODELS_DIRECTORY_PATH, "XiYanSQL"), torch_dtype=torch.bfloat16,
            device_map="auto")
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def _answer_single_question(self, question):
        return self.pipe(
            question, 
            return_full_text=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=1024,
            temperature=0.1,
            top_p=0.8,
            do_sample=True
        )[0]['generated_text']
    
    def _answer_single_prompt_with_transition_scores(self, prompt):
        message = [{'role': 'user', 'content': prompt}]
        text = self.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **model_inputs,
            pad_token_id = self.tokenizer.pad_token_id,
            eos_token_id = self.tokenizer.eos_token_id,
            max_new_tokens = 1024,
            temperature = 0.1,
            top_p = 0.8,
            do_sample = True,
            return_dict_in_generate = True,
            output_scores = True
        )

        transition_scores = self.model.compute_transition_scores(
            outputs.sequences,
            outputs.scores,
            normalize_logits = True
        )

        generated_tokens = outputs.sequences[:, len(model_inputs.input_ids)]
    
        for tok, score in zip(generated_tokens[0], transition_scores[0]):
        # | token | token string | log probability | probability
            print(f"| {tok:5d} | {self.tokenizer.decode(tok):8s} | {score.numpy():.3f} | {np.exp(score.numpy()):.2%}")

class DeepSeekQwenModel(NL2SQLModel):
    def __init__(self, connection, benchmark_set, prompt_strategy):
        super().__init__(connection, benchmark_set, prompt_strategy)
        self.tokenizer = AutoTokenizer.from_pretrained(join(MODELS_DIRECTORY_PATH, "DeepSeekQwen"))
        self.model = AutoModelForCausalLM.from_pretrained(join(MODELS_DIRECTORY_PATH, "DeepSeekQwen"), torch_dtype=torch.bfloat16,
            device_map="auto")
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def _answer_single_question(self, question):
        return self.pipe(
            question, 
            return_full_text=False,
            max_new_tokens=1024,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            temperature=0.6
        )[0]['generated_text']
    
    def _prune_generated_query(self, query):
        query = super()._prune_generated_query(query)
        query = re.sub(r".*SELECT", "SELECT", query, flags=re.IGNORECASE)
        return re.sub(r'([a-z])([A-Z])', r'\1 \2', query)
    
class DeepSeekLlamaModel(NL2SQLModel):
    def __init__(self, connection, benchmark_set, prompt_strategy):
        super().__init__(connection, benchmark_set, prompt_strategy)
        self.tokenizer = AutoTokenizer.from_pretrained(join(MODELS_DIRECTORY_PATH, "DeepSeekLlama"))
        self.model = AutoModelForCausalLM.from_pretrained(join(MODELS_DIRECTORY_PATH, "DeepSeekLlama"), torch_dtype=torch.bfloat16,
            device_map="auto")
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def _answer_single_question(self, question):
        return self.pipe(
            question, 
            return_full_text=False,
            max_new_tokens=1024,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            temperature=0.6
        )[0]['generated_text']
    
class LlamaModel(NL2SQLModel):
    def __init__(self, connection, benchmark_set, prompt_strategy):
        super().__init__(connection, benchmark_set, prompt_strategy)
        self.tokenizer = AutoTokenizer.from_pretrained(join(MODELS_DIRECTORY_PATH, "Llama"))
        self.model = AutoModelForCausalLM.from_pretrained(join(MODELS_DIRECTORY_PATH, "Llama"), torch_dtype=torch.bfloat16,
            device_map="auto")
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def _answer_single_question(self, question):
        return self.pipe(
            question, 
            return_full_text=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=1024,
        )[0]['generated_text']
    
    def _prune_generated_query(self, query: str):
        if ';' in query:
            return super()._prune_generated_query(query)
        else:
            return super()._prune_generated_query(query) + ";"