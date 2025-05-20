# Parts of the code in this file is taken from the dts_sql repo: https://github.com/MohammadrezaPourreza/DTS-SQL

import torch
import re
import sqlite3
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria
from tqdm import tqdm
import json
import statistics

NUMBER_OF_TRIALS = 10
EVALUATION_DATA_PATH = ".local/SchemaLinker/spider_exsl_all_to_single_test.json"
BASE_DABATASES_DIR =  "DTS-SQL/test_database/"
OUTPUT_DIR = ".local/experiments/schema_linking/spider/dts_sql/"

schema_linker_adapter_path = "MrezaPRZ/DeepSchema_BIRD"

#loading the models
schema_model = AutoModelForCausalLM.from_pretrained(schema_linker_adapter_path,attn_implementation="flash_attention_2",torch_dtype=torch.bfloat16,device_map = "auto")
tokenizer = AutoTokenizer.from_pretrained(schema_linker_adapter_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

class EosListStoppingCriteriaSchema(StoppingCriteria):
    def __init__(self, eos_sequence = [6204, 185, 10897]):
        self.eos_sequence = eos_sequence

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_ids = input_ids[:,-len(self.eos_sequence):].tolist()
        return self.eos_sequence in last_ids

def remove_spaces(text):
  return re.sub(r'\s+', ' ', text)

def extract_db_id(m_schema: str) -> str:
    match = re.search(r'【DB_ID】\s*(\w+)', m_schema)
    return match.group(1) if match else None

def get_all_table_names(db_uri: str) -> list[str]:
    conn = sqlite3.connect(db_uri)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    table_names = cursor.fetchall()
    conn.close()
    return [table_name[0] for table_name in table_names]

def get_table_schema_with_samples(
    db_uri: str, table_name: str, sample_limit: int = 0
) -> str:
    conn = sqlite3.connect(db_uri)
    cursor = conn.cursor()

    # Fetch table schema
    cursor.execute(f"PRAGMA table_info(`{table_name}`);")
    columns = cursor.fetchall()
    cursor.execute(f"PRAGMA foreign_key_list(`{table_name}`);")
    foreign_keys = cursor.fetchall()
    cursor.execute(f"PRAGMA index_list(`{table_name}`);")
    primary_key_indices = cursor.fetchall()
    primary_key_columns = []

    for index_info in primary_key_indices:
        index_name = index_info[1]
        cursor.execute(f"PRAGMA index_info(`{index_name}`);")
        index_columns = cursor.fetchall()
        primary_key_columns.extend(column[2] for column in index_columns)

    # Construct CREATE TABLE statement
    schema_str = f"CREATE TABLE `{table_name}` (\n"
    for column in columns:
        column_name = column[1]
        data_type = column[2]
        schema_str += f"  {column_name} {data_type}"
        if column_name in primary_key_columns:
            schema_str += " PRIMARY KEY"
        for foreign_key in foreign_keys:
            if column_name == foreign_key[3]:
                schema_str += f" REFERENCES {foreign_key[2]}({foreign_key[4]})"

        schema_str += ",\n"
    schema_str = schema_str.rstrip(",\n")
    schema_str += "\n);\n"

    
    cursor.execute(f"SELECT * FROM `{table_name}` LIMIT {sample_limit};")
    sample_rows = cursor.fetchall()

    if len(sample_rows) > 0:
        schema_str += f"Sample rows from `{table_name}`:\n"
        for row in sample_rows:
            formatted_row = ", ".join(str(item) for item in row)
            schema_str += f"{formatted_row}\n"

    conn.close()
    return schema_str

def generate_schema(inputs, merged_model):
  output_tokens = merged_model.generate(inputs, max_new_tokens=250, do_sample=False, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id, stopping_criteria = [EosListStoppingCriteriaSchema()])
  return tokenizer.decode(output_tokens[0][len(inputs[0]):], skip_special_tokens=True)

def get_relevant_tables(db_uri, question):
    table_names = get_all_table_names(db_uri)
    database_schema = ""
    for table_name in table_names:
        schema = get_table_schema_with_samples(db_uri, table_name, 0)
        database_schema += schema + "\n"
    user_message = f"""Given the following SQL tables, your job is to determine the columns and tables that the question is referring to.
{database_schema}
####
Question: {question}
"""
    messages = [
        {"role": "user", "content": user_message.strip()}
    ]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True,tokenize = True).to(schema_model.device)
    response = generate_schema(inputs, schema_model)
    if "Tables: " in response:
        response = response.split("Tables: ")[1]
    if ";" in response:
        response = response.split(";")[0]
    schema_linking_tables = re.sub(r'\s+', ' ', response).strip()
    schema_linking_tables = schema_linking_tables.split(", ")
    for table in schema_linking_tables:
        table = table.replace("**", "").replace("--", "").replace("'","").strip()
        table = table.lower()
    return schema_linking_tables

def extract_db_id(mschema: str) -> str:
    """
    Extracts the DB_ID from an mschema string.

    Args:
        mschema (str): The mschema string.

    Returns:
        str: The extracted DB_ID or None if not found.
    """
    match = re.search(r'【DB_ID】\s*(\w+)', mschema)
    if match:
        return match.group(1)
    return None


if __name__ == "__main__":
    table_recall_results = []
    table_precision_results = []

    for trial_num in tqdm(range(NUMBER_OF_TRIALS)):
        with open(EVALUATION_DATA_PATH, "r") as eval_file:
            eval_set = json.load(eval_file)

        report = []
        recall_sum = 0
        precision_sum = 0
        for example in tqdm(eval_set):
            db_id = extract_db_id(example["schema"])
            db_uri = f"{BASE_DABATASES_DIR}{db_id}/{db_id}.sqlite"

            goal_tables = {column.split(" ")[0] for column in example["goal answer"]}
            predictions = get_relevant_tables(db_uri, example["question"])
            correct_predictions = [table for table in predictions if table in goal_tables]

            recall = len(correct_predictions) / len(goal_tables)
            precision = len(correct_predictions) / len(predictions)
            
            recall_sum += recall
            precision_sum += precision

            report.append({"question": example["question"], "goal tables": list(goal_tables), "predictions": predictions, "precision": precision, "recall": recall})

        report.append({"Dataset size": len(eval_set), "Total precision": precision_sum / len(eval_set), "Total recall": recall_sum / len(eval_set)})
        table_recall_results.append(report[-1]["Total recall"])
        table_precision_results.append(report[-1]["Total precision"])

        with open(f"{OUTPUT_DIR}trial_{trial_num}.json", "w") as file:
            json.dump(report, file, indent=4)

    overall_report = {
        "Table recall": {"mean": statistics.mean(table_recall_results), "standard deviation": statistics.stdev(table_recall_results)},
        "Table precision": {"mean": statistics.mean(table_precision_results), "standard deviation": statistics.stdev(table_precision_results)}
    }

    with open(f"{OUTPUT_DIR}overview.json", "w") as file:
        json.dump(overall_report, file, indent=4)