from transformers import AutoTokenizer, AutoModelForMaskedLM
from datasets import Dataset
import json


TRAIN_DATA_FILEPATH = ""


def train():
    dataset = _load_filtered_dataset(TRAIN_DATA_FILEPATH)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    


def _load_filtered_dataset(filepath):
    with open(filepath, 'r') as fp:
        raw_data = json.load(fp)

    filtered_data = []
    for entry in raw_data:
        if entry.get("feasible") is not None:
            filtered_data.append({
                "question": entry.get("question"),
                "schema": entry.get("schema"),
                "label": int(entry.get("feasible"))
            })

    return Dataset.from_list(filtered_data)
