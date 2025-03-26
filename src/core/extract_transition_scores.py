from transformers import AutoTokenizer, AutoModelForCausalLM
from os.path import join
import torch
import numpy as np

MODELS_DIRECTORY_PATH = "models/"
PATH = join(MODELS_DIRECTORY_PATH, "XiYanSQL")

tokenizer = AutoTokenizer.from_pretrained(PATH)
model = AutoModelForCausalLM.from_pretrained(PATH,  torch_dtype=torch.bfloat16, device_map="auto")

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

# Concatenating the input
formatted_input = schema + "\n\n" + query + "\n\n"

message = [{'role': 'user', 'content': formatted_input}]
text = tokenizer.apply_chat_template(
    message,
    tokenize=False,
    add_generation_prompt=True
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
outputs = model.generate(
    **model_inputs,
    pad_token_id = tokenizer.pad_token_id,
    eos_token_id = tokenizer.eos_token_id,
    max_new_tokens = 1024,
    temperature = 0.1,
    top_p = 0.8,
    do_sample = True,
    return_dict_in_generate = True,
    output_scores = True,
    return_legacy_cache=True
)

transition_scores = model.compute_transition_scores(
    outputs.sequences,
    outputs.scores,
    normalize_logits = True
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, outputs.sequences)
]

print("| token | token string | log probability | probability")
for tok, score in zip(generated_ids[0], transition_scores[0]):
    print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy():.3f} | {np.exp(score.numpy()):.2%}")