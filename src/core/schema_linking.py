from transformers import AutoTokenizer, AutoModel
from os.path import join
import torch
import torch.nn.functional as F

MODELS_DIRECTORY_PATH = "models/"
PATH = join(MODELS_DIRECTORY_PATH, "XiYanSQL")

tokenizer = AutoTokenizer.from_pretrained(PATH)
model = AutoModel.from_pretrained(PATH,  torch_dtype=torch.bfloat16, device_map="auto")

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

query = "Show the titles of books in descending order of publication price."

columns = [
    "« publication publication_id »",
    "« publication book_id »",
    "« publication publisher »",
    "« publication publication_date »",
    "« publication price »",
    "« book book_id »",
    "« book title »",
    "« book issues »",
    "« book writer »",
]

# Concatenating the input
formatted_input = schema + "\n\n" + query + "\n\n" + " ".join(columns)

# Tokenize input
inputs = tokenizer(formatted_input, return_tensors="pt", truncation=True, padding=True)#.to("cuda")

with torch.no_grad():
    outputs = model(**inputs)

hidden_states = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_dim)

# Identify positions of `«` and `»`
input_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
alpha_positions = [i for i, token in enumerate(input_tokens) if token == "ĠÂ«" or token == "Â«"]
omega_positions = [i for i, token in enumerate(input_tokens) if token == "ĠÂ»"]

# Extract corresponding embeddings
alpha_embeddings = hidden_states[0, alpha_positions, :]  # (num_columns, hidden_dim)
omega_embeddings = hidden_states[0, omega_positions, :]  # (num_columns, hidden_dim)

# Concatenate the embeddings
column_embeddings = torch.cat([alpha_embeddings, omega_embeddings], dim=-1)  # (num_columns, 2*hidden_dim)


weights_final = model.lm_head.weight # Extract weights from the final layer of the model

relevance_scores = None # TODO: Implement relevance scores
# Apply sigmoid to get probabilities
relevance_probs = torch.sigmoid(relevance_scores)

# Print relevance scores for each column
for col, score in zip(columns, relevance_probs.tolist()):
    print(f"{col}: {score}")