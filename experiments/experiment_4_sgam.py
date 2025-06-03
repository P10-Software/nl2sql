from src.core.extractive_schema_linking import load_schema_linker, get_focused_schema
from src.core.schema_chunking import chunk_mschema
from src.core.abstention_classifier import load_abstention_classifier
import json
from tqdm import tqdm

with open(".local/metadata_reliability_natural.json", "r") as file:
    dataset = json.load(file)

with open(".local/mschema_trial_metadata_natural.txt", "r") as file:
    schema = file.read()

schema_linker = load_schema_linker()
chunks = chunk_mschema(schema, schema_linker.tokenizer, False, k=1)
abstention_classfier = load_abstention_classifier()

results = []
for example in tqdm(dataset):
    focused_schema = get_focused_schema(schema_linker, example["question"], chunks, schema)
    abstention_prediction = abstention_classfier.classify(example["question"], focused_schema)
    results.append({"question": example["question"], "golden_query": example["golden_query"], "focused_schema": focused_schema, "abstention_prediction": abstention_prediction})

with open(".local/experiments/full_pipeline/metadata_sgam_results.json", "w") as file:
    json.dump(results, file, indent=4)