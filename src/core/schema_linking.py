from transformers import AutoTokenizer, AutoModelForCausalLM
from os.path import join
import torch
import numpy as np
from json import dump

MODELS_DIRECTORY_PATH = "models/"
PATH = join(MODELS_DIRECTORY_PATH, "XiYanSQL")

class SchemaExtractor():
    def __init__(self, schema):
        self.tokenizer = AutoTokenizer.from_pretrained(PATH)
        self.model = AutoModelForCausalLM.from_pretrained(PATH,  torch_dtype=torch.bfloat16, device_map="auto")
        self.schema = schema
        self.prompt_strategy = """
            You are now a {SQL_DIALECT} data analyst, and you are given a database schema as follows:

            【Schema】
            {SCHEMA}

            【Question】
            {NL_QUESTION}

            【Evidence】
            {EVIDENCE}

            Please read and understand the database schema carefully, and generate the relevant schema elements based on the user's question and evidence. Do not generate SQL. The generated schema is protected by ```schema and ```
        """

    def generate_relevant_schema_elements(self, question) -> dict:
        prompt = self.prompt_strategy.format(SQL_DIALECT="sqlite", SCHEMA=self.schema, NL_QUESTION=question, EVIDENCE="")
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
            output_scores = True,
            return_legacy_cache=True
        )

        transition_scores = self.model.compute_transition_scores(
            outputs.sequences,
            outputs.scores,
            normalize_logits = True
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, outputs.sequences)
        ]

        relevant_schema_with_probs = []
        for tok, score in zip(generated_ids[0], transition_scores[0]):
            relevant_schema_with_probs.append({"token": f"{tok:5d}", "token_string": f"{self.tokenizer.decode(tok):8s}", "log_probability": f"{score.numpy():.3f}", "probability": f"{np.exp(score.numpy()):.2%}"})

        return relevant_schema_with_probs
    
if __name__ == "__main__":
    schema = """
    CREATE TABLE publication (
        publication_id NUMBER PRIMARY KEY,
        book_id NUMBER,
        publisher TEXT,
        publication_date TEXT,
        price NUMBER,
        FOREIGN KEY(book_id) REFERENCES book(book_id)
    );
    CREATE TABLE book (
        book_id NUMBER PRIMARY KEY,
        title TEXT,
        issues NUMBER,
        writer TEXT
    );
    """

    question = "Show the titles of books in descending order of publication price."
    schema_extractor = SchemaExtractor(schema)

    with open("transition_scores_test.json", "w") as file:
        dump(schema_extractor.generate_relevant_schema_elements(question), file, indent=4)