from transformers import AutoTokenizer, AutoModel
from os.path import join
import torch
import re

MODELS_DIRECTORY_PATH = "models/"
PATH = join(MODELS_DIRECTORY_PATH, "XiYanSQL")

class SchemaLinker():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(PATH)
        self.model = AutoModel.from_pretrained(PATH,  torch_dtype=torch.bfloat16, device_map="auto")

    def get_relevance_for_schema_chunk(self, schema_chunk: str, question: str):
        columns = self._get_columns(schema_chunk)
        formatted_input = schema + "\n\n" + question + "\n\n" + " ".join(columns)

        # Tokenize input
        inputs = self.tokenizer(formatted_input, return_tensors="pt", truncation=True, padding=True)#.to("cuda")

        with torch.no_grad():
            outputs = self.model(**inputs)

        hidden_states = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_dim)

        # Identify positions of `«` and `»`
        input_tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        alpha_positions = [i for i, token in enumerate(input_tokens) if token == "ĠÂ«" or token == "Â«"]
        omega_positions = [i for i, token in enumerate(input_tokens) if token == "ĠÂ»"]

        # Extract corresponding embeddings
        alpha_embeddings = hidden_states[0, alpha_positions, :]  # (num_columns, hidden_dim)
        omega_embeddings = hidden_states[0, omega_positions, :]  # (num_columns, hidden_dim)

        # Concatenate the embeddings
        column_embeddings = torch.cat([alpha_embeddings, omega_embeddings], dim=-1)  # (num_columns, 2*hidden_dim)


        weights_final = self.model.lm_head.weight # Extract weights from the final layer of the model

        relevance_scores = None # TODO: Implement relevance scores
        # Apply sigmoid to get probabilities
        relevance_probs = torch.sigmoid(relevance_scores)

        return zip(columns, relevance_probs.tolist())
    
    def _get_columns(schema_chunk: str):
        columns_in_schema = ""

        # Split schema by tables
        table_sections = re.split(r"# Table: (\w+)", schema_chunk)[1:]  
        for i in range(0, len(table_sections), 2):
            table_name = table_sections[i].strip()  
            columns_section = table_sections[i + 1]

            # Extract column names
            column_matches = re.findall(r"\(\s*(\w+):", columns_section)
            for column in column_matches:
                columns_in_schema += f"<< {table_name} {column} >>\n"

        return columns_in_schema

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

    schema_linker = SchemaLinker()
    column_probs = schema_linker.get_relevance_for_schema_chunk(schema, question)

    for col, score in column_probs:
        print(f"{col}:{score}")

