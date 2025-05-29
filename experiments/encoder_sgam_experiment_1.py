from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.core.extractive_schema_linking import load_schema_linker, get_focused_schema
from src.core.schema_chunking import chunk_mschema
from tqdm import tqdm
import json
import torch

SCHEMA_LINKER_PATH = "models/EXSL/OmniSQL_7B_rmc_efficiency_schema_linker_trial_39.pth"
ABSTENTION_MODEL_PATH  = ".local/AbstentionClassifier/encoder_best/checkpoint-2450/"
TEST_SET_PATH = ".local/bird_abstention_eval_set.json"
RESULT_PATH = ".local/experiments/abstention/bird/sgam_encoder.json"

with open(TEST_SET_PATH, "r") as file:
    test_set = json.load(file)

abstention_tokenizer = AutoTokenizer.from_pretrained(ABSTENTION_MODEL_PATH, device_map="cuda")
abstention_model = AutoModelForSequenceClassification.from_pretrained(ABSTENTION_MODEL_PATH, num_labels=2, device_map="cuda")
abstention_model.eval()
schema_linker = load_schema_linker(SCHEMA_LINKER_PATH)

result = []
true_pos = 0
false_pos = 0
true_neg = 0
false_neg = 0

for example in tqdm(test_set):
    chunks = chunk_mschema(example["schema"], schema_linker.tokenizer, False, k=1)
    focused_schema = get_focused_schema(schema_linker, example["question"], chunks, example["schema"], threshold=0.15)

    inputs = abstention_tokenizer(example["question"], focused_schema, return_tensors="pt", truncation=True, padding="max_length", max_length=8192).to("cuda")

    # Run inference
    with torch.no_grad():
        outputs = abstention_model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()

    if prediction == 0:
        if example["label"] == 0:
            true_pos += 1
        else:
            false_pos += 1
    else:
        if example["label"] == 0:
            false_neg += 1
        else:
            true_neg += 1
    
    result.append({"question": example["question"], "goal feasibility": example["label"], "predicted feasibility": prediction})

precision = true_pos / (true_pos + false_pos)
recall = true_pos / (true_pos + false_neg)
f2 = (5 * precision * recall) / ((4 * precision) + recall)
result.append({"precision": precision, "recall": recall, "f2": f2})

with open(RESULT_PATH, "w") as file:
    json.dump(result, file, indent=4)