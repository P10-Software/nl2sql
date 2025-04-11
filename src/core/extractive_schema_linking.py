from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from os.path import join
import torch
import re
from transformer_heads.config import HeadConfig
from transformer_heads.util.helpers import get_model_params
from transformer_heads import load_headed
from json import load

MODELS_DIRECTORY_PATH = "models/"
MODEL_PATH = join(MODELS_DIRECTORY_PATH, "SQLCoder")
HEAD_PATH = join(MODELS_DIRECTORY_PATH, "EXSL")

class SchemaLinker():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model_params = get_model_params(MODEL_PATH)
        self.model = load_headed(
            model_params["model_class"],
            MODEL_PATH,
            head_folder_path=HEAD_PATH,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )

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
    
def inject_and_train_new_head():
    model_params = get_model_params(MODEL_PATH)
    model_class = model_params["model_class"]
    hidden_size = model_params["hidden_size"]
    vocab_size = model_params["vocab_size"]

    relevance_head = HeadConfig(
        name="relevance_head",
        layer_hook=-1,
        in_size=hidden_size,
        output_activation="linear",
        is_causal_lm=True,
        loss_fct="cross_entropy",
        num_outputs=vocab_size,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_headed(
        model_class,
        MODEL_PATH,
        [relevance_head],
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    # TODO: Traing new head
    with open(".local/spider_exsl_train.json", "r") as file:
        train_raw = load(file)

    train_set = RelevanceDataset(train_raw, tokenizer)

    args = TrainingArguments(
        output_dir=HEAD_PATH,
        learning_rate=0.000005,
        num_train_epochs=1,
        logging_steps=100,
        do_eval=False,
        remove_unused_columns=False,
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model,
        args=args,
        train_dataset=train_set,
        data_collator=collator,
        
    )
    trainer.train()

    model.save_pretrained(HEAD_PATH)

class RelevanceDataset(torch.Dataset):
    def __init__(self, raw_data, tokenizer, max_length=512):
        self.examples = []
        self.tokenizer = tokenizer

        for example in raw_data:
            input_str = example["input"]
            columns = self.extract_columns(input_str)
            labels = self.create_labels(columns, example["goal answer"])

            tokenized = tokenizer(
                input_str,
                truncation=True,
                padding='max_length',
                max_length=max_length
            )

            tokenized["labels"] = torch.tensor(labels, dtype=torch.float)
            self.examples.append(tokenized)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {key: torch.tensor(val) for key, val in self.examples[idx].items()}

    def extract_columns(self, input_str):
        columns_in_schema = []

        # Split schema by tables
        table_sections = re.split(r"# Table: (\w+)", input_str)[1:]  
        for i in range(0, len(table_sections), 2):
            table_name = table_sections[i].strip()  
            columns_section = table_sections[i + 1]

            # Extract column names
            column_matches = re.findall(r"\(\s*(\w+):", columns_section)
            for column in column_matches:
                columns_in_schema.append(f"{table_name} {column}")
        return columns_in_schema

    def create_labels(self, columns, goal_answer):
        goal_lower = [g.lower() for g in goal_answer]
        return [1 if col.lower() in goal_lower else 0 for col in columns]

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

