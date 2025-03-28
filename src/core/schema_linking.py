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
/* Database schema */
{SCHEMA}

Attention:
1. if the question have when\where\which, pay attention to pick table.column related to time, location and name in #columns
2. Please answer the question in the following format without any other content:
```
#columns: The top 10 columns relevant to the question( format: table.column_1, table.column_2 ...)
#values: Potential filter values that the question might query(format: "value1", "value2" ...)
```
/* Answer the following: {NL_QUESTION} */
        """

    def generate_relevant_schema_elements(self, question) -> dict:
        prompt = self.prompt_strategy.format(SCHEMA=self.schema, NL_QUESTION=question)
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
        ).cpu()

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, outputs.sequences)
        ]

        is_columns = False
        new_candidate = []
        relevant_schema_with_probs = {"columns": [], "values": []}

        for tok, score in zip(generated_ids[0], transition_scores[0]):
            tok_string = self.tokenizer.decode(tok).strip()
            match tok_string:
                case "#":
                    if new_candidate:
                        relevant_schema_with_probs["columns" if is_columns else "values"].append(new_candidate)
                        new_candidate = []
                case "columns":
                    is_columns = True
                case "values":
                    is_columns = False
                case ",":
                    relevant_schema_with_probs["columns" if is_columns else "values"].append(new_candidate)
                    new_candidate = []
                case "\n" | ":" | "":
                    continue
                case _:
                    new_candidate.append({tok_string: f"{np.exp(score.numpy()):.2%}"})

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