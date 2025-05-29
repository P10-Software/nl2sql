import json
from sklearn.metrics import precision_recall_fscore_support
from src.core.schema_chunking import chunk_mschema
from src.core.extractive_schema_linking import get_focused_schema, load_schema_linker
from src.core.abstention_classifier import AbstentionClassifier, load_abstention_classifier
from src.common.logger import get_logger
from tqdm import tqdm

logger = get_logger(__name__)

BIRD_TEST_DATA = '.local/bird_abstention_eval_set.json'  # Doesnt need DB, schema already part of test-data
EHRSQL_MIMIC_TEST_DATA = ''
EHRSQL_MIMIC_MSCHEMA = ''
EHRSQL2_TEST_DATA = ''
EHRSQL2_DB = ''
TRIAL_TEST_DATA = '.local/metadata_reliability_natural.json'
TRAIL_MSCHEMA = '.local/mschema_trial_metadata_natural.txt'
MODEL_PATH = '.local/AbstentionClassifier/BinaryHead/best_classifier.pt'
LINKER_PATH = 'models/EXSL/OmniSQL_7B_rmc_efficiency_schema_linker_trial_39.pth'
LINKER_THRESHOLD = 0.15


def run_experiment(dataset, dataset_name):
    schema_linker = load_schema_linker(LINKER_PATH)
    abstention_model = _load_model()

    relations = True if dataset_name == 'TrialBench' else False

    y_true = []
    y_pred = []

    for example in tqdm(dataset):
        chunks = chunk_mschema(mschema=example['schema'], tokenizer=schema_linker.tokenizer, with_relations=relations, k=1)
        focused_schema = get_focused_schema(schema_linker=schema_linker, question=example['question'], chunks=chunks, schema=example['schema'], threshold=LINKER_THRESHOLD)
        classification = abstention_model.classify(example['question'], schema=focused_schema)

        y_true.append(example['feasible'])
        y_pred.append(1 if classification == 'feasible' else 0)

    precision, recall, f2, _ = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, average='binary', pos_label=0, beta=2)
    logger.info(f'Finished decoder abstention for dataset: {dataset_name}, with results: precision: {precision}, recall: {recall}, f2: {f2}')
    results = {
        'y_true': y_true,
        'y_pred': y_pred,
        'precision': precision,
        'recall': recall,
        'f2': f2
    }

    return results


def prep_ehrsql():
    new_data = []
    schema = _load_mschema(EHRSQL_MIMIC_MSCHEMA)
    with open(EHRSQL_MIMIC_TEST_DATA, 'r') as fp:
        data = json.load(fp)
    for example in data:
        if example['goal_query'] is None:
            new_data.append({
                "id": example['id'],
                "question": example['question'],
                "feasible": 0,
                "schema": schema
            })
        else:
            new_data.append({
                "id": example['id'],
                "question": example['question'],
                "feasible": 1,
                "schema": schema
            })

    return new_data


def prep_trial():
    new_data = []
    schema = _load_mschema(TRAIL_MSCHEMA)
    with open(TRIAL_TEST_DATA, 'r') as fp:
        data = json.load(fp)
    for example in data:
        if example['goal_query'] is None:
            new_data.append({
                "id": example['id'],
                "question": example['question'],
                "feasible": 0,
                "schema": schema
            })
        else:
            new_data.append({
                "id": example['id'],
                "question": example['question'],
                "feasible": 1,
                "schema": schema
            })

    return new_data


def _load_mschema(path: str):
    with open(path, 'r') as fp:
        file = fp.read()
    return file


def _load_model():
    return load_abstention_classifier(MODEL_PATH)


if __name__ == '__main__':
    results = []

    logger.info('Running first experiment: Decoder on BIRD...')
    with open(BIRD_TEST_DATA, 'r') as fp:
        bird_data = json.load(fp)
    bird_res = run_experiment(bird_data, 'BIRD')
    results.append(bird_res)

    logger.info('Running second experiment: Decoder on TrialBench...')
    trial_data = prep_trial()
    trial_res = run_experiment(trial_data, 'TrialBench')
    results.append(trial_res)

    with open('./local/experiments/abstention/decoder_pooling_all.json', 'w') as fp:
        json.dump(results)