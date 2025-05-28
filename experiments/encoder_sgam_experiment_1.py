from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import json

T5_PATH = ".local/trust_sql/t5_sqlgen/checkpoint-12723/"
TEST_SET_PATH = ".local/bird_abstention_eval_set.json"
RESULT_PATH = ".local/experiments/abstention/bird/trust_pipe.json"

with open(TEST_SET_PATH, "r") as file:
    test_set = json.load(file)

t5_tokenizer = AutoTokenizer.from_pretrained(T5_PATH)
t5_model = AutoModelForSeq2SeqLM.from_pretrained(T5_PATH, device_map="auto")

result = []
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