from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from tqdm import tqdm
from peft import PeftModel
import json
import torch

T5_PATH = ".local/trust_sql/t5_sqlgen/checkpoint-12723/"
SQL_CODER_TOKENIZER_PATH = ".local/trust_sql/sql_coder_infeasible/tokenizer/"
PRE_ABSTENTION_PATH = ".local/trust_sql/sql_coder_infeasible/final/"
POST_ABSTENTION_PATH = ".local/trust_sql/sql_coder_error/final/"
TEST_SET_PATH = ".local/bird_abstention_eval_set.json"
RESULT_PATH = ".local/experiments/abstention/bird/trust_pipe.json"
PRE_FEASIBLE_PATH = ".local/experiments/abstention/pre_feasible_dataset_trust.json"

with open(TEST_SET_PATH, "r") as file:
    test_set = json.load(file)

t5_tokenizer = AutoTokenizer.from_pretrained(T5_PATH)
t5_model = AutoModelForSeq2SeqLM.from_pretrained(T5_PATH, device_map="auto")

sqlcoder_tokenizer = AutoTokenizer.from_pretrained(SQL_CODER_TOKENIZER_PATH)

base_model_1 = AutoModelForCausalLM.from_pretrained("defog/sqlcoder-7b-2", device_map="auto", torch_dtype=torch.bfloat16)
base_model_1.resize_token_embeddings(len(sqlcoder_tokenizer))
pre_abstention_model = PeftModel.from_pretrained(base_model_1, PRE_ABSTENTION_PATH)

base_model_2 = AutoModelForCausalLM.from_pretrained("defog/sqlcoder-7b-2", device_map="auto", torch_dtype=torch.bfloat16)
base_model_2.resize_token_embeddings(len(sqlcoder_tokenizer))
post_abstention_model  = PeftModel.from_pretrained(base_model_2, POST_ABSTENTION_PATH)

def get_t5_prompt(question, schema):
    start_prompt = "Database:\n"
    middle_prompt = "\n\nQuestion:\n"
    end_prompt = "\n\nAnswer:\n"
    return start_prompt + schema + middle_prompt + question + end_prompt

def get_pre_abstention_prompt(question, schema):
    return f"""
        ### Task
        Generate a SQLite SQL query to answer [QUESTION]{question}[/QUESTION]

        ### Instructions 
        - If you cannot answer the question with the available database schema, return 'I do not know'

        ### Database Schema
        The query will run on a database with the following schema:
        {schema}

        ### Answer
        Given the database schema, here is the SQL query that [QUESTION]{question}[/QUESTION]
        [SQL]
"""

def get_error_detect_prompt(schema, question, sql):
    return f"""
        #### Based on the question and predicted SQLite SQL, are you sure the SQL below is correct? If you consider the SQL is correct, answer me with 'correct'. 
        If not, answer me with 'incorrect'. Only output your response without explanation.

        ### Database Schema
        The query will run on a database with the following schema:
        {schema}

        Question: {question}
        Predicted SQL: {sql}
        Answer:        
"""

result = []
pre_feasible_set = []
true_pos = 0
false_pos = 0
true_neg = 0
false_neg = 0

for example in tqdm(test_set):
    cls_prompt = get_pre_abstention_prompt(example["question"], example["schema"])
    inputs = sqlcoder_tokenizer(cls_prompt, return_tensors="pt").to(pre_abstention_model.device)
    input_length = inputs['input_ids'].shape[1] 
    outputs = pre_abstention_model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        pad_token_id=sqlcoder_tokenizer.eos_token_id  # Ensure padding token is defined
    )
    generated_tokens = outputs[0][input_length:]  # Skip prompt
    pre_classification = sqlcoder_tokenizer.decode(generated_tokens, skip_special_tokens=True)
    if "I do not know" in pre_classification.strip():
        result.append({"question": example["question"], "goal feasibility": example["label"], "predicted feasibility": 0})

        if example["label"] == 0:
            true_pos += 1
        else:
            false_pos += 1

        continue
    else:
        pre_feasible_set.append(example)

    sql_prompt = get_t5_prompt(example["question"], example["schema"])
    inputs = t5_tokenizer(sql_prompt, return_tensors="pt").to(t5_model.device)
    outputs = t5_model.generate(**inputs, max_new_tokens=200)
    sql = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

    error_prompt = get_error_detect_prompt(example["schema"], example["question"], sql)
    inputs = sqlcoder_tokenizer(error_prompt, return_tensors="pt").to(post_abstention_model.device)
    outputs = post_abstention_model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False,
        pad_token_id=sqlcoder_tokenizer.eos_token_id
    )
    input_len = inputs['input_ids'].shape[1]
    generated_tokens = outputs[0][input_len:]
    post_classification = sqlcoder_tokenizer.decode(generated_tokens, skip_special_tokens=True)

    if  "incorrect" in post_classification.strip():
        feasibility_prediction = 0

        if example["label"] == 0:
            true_pos += 1
        else:
            false_pos += 1
    else:
        feasibility_prediction = 1

        if example["label"] == 0:
            false_neg += 1
        else:
            true_neg += 1
    
    result.append({"question": example["question"], "goal feasibility": example["label"], "predicted feasibility": feasibility_prediction})

precision = true_pos / (true_pos + false_pos)
recall = true_pos / (true_pos + false_neg)
f2 = (5 * precision * recall) / ((4 * precision) + recall)
result.append({"precision": precision, "recall": recall, "f2": f2})

with open(RESULT_PATH, "w") as file:
    json.dump(result, file, indent=4)

with open(PRE_FEASIBLE_PATH, "w") as file:
    json.dump(pre_feasible_set, file, indent=4)