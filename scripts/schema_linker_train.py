import torch
import optuna
import json
from tqdm import tqdm
from src.common.logger import get_logger
from src.core.extractive_schema_linking import ExSLcModel, prepare_input, predict_relevance_coarse, load_schema_linker

TRAIN_SET_PATH=".local/SchemaLinker/spider_exsl_all_to_single_train.json"
EVAL_SET_PATH=".local/SchemaLinker/spider_exsl_dev.json"
K=5
RESULT_DIR_PATH=f".local/SchemaLinker/OmniSQL7B_optuna/"
BASE_MODEL="seeklhy/OmniSQL-7B"
NUMBER_OF_TRIALS=50

logger = get_logger(__name__)

def run_study():
    """
    Creates or resumes an Optuna study and launches hyperparameter
    optimization.
    """
    study = optuna.create_study(
        direction="minimize",
        study_name="schema_linker_study_nrmc",
        storage="sqlite:///optuna_study_schema_linker_omnisql_nrmc.db",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=NUMBER_OF_TRIALS)
    logger.info(f"Best trial: {study.best_trial}")

def objective(trial: optuna.trial):
    model = ExSLcModel(BASE_MODEL)
    model.to("cuda")

    config = {
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
        "learning_rate": trial.suggest_float("learning_rate", 5e-7, 5e-5, log=True),
        "epochs": trial.suggest_int("epochs", 2, 4)
    }

    logger.info(
        f"Running trial {trial.number}, using parameters: {trial.params}")

    # Load train data set
    with open(TRAIN_SET_PATH, "r") as train_file:
        train_set = json.load(train_file)

    trained_model = train_coarse_grained(model, train_set, config)

    # Load eval data and evaluate
    with open(EVAL_SET_PATH, "r") as eval_file:
        eval_set = json.load(eval_file)

    eval_result = evaluate_coarse_grained(trained_model, eval_set, K)

    with open(RESULT_DIR_PATH + f"table_n_rmc_spider_exsl_recall_at_{K}_trial_{trial.number}.json", "w") as result_file:
        json.dump(eval_result, result_file, indent=4)
    logger.info(eval_result[-1])

    return eval_result[-1]["Average table-n-RMC"]

def train_coarse_grained(model, train_data, config):
    # Initialize Accelerator

    # Prepare dataset
    train_dataset = SchemaLinkingDatasetCoarse(train_data)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    
    # Optimizer with paper's parameters
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )

    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Training loop
    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        
        for example in tqdm(train_dataloader):
            optimizer.zero_grad()

            labels = example["labels"].squeeze(0).to("cuda")
            
            # Forward pass
            logits = model(example["input"][0])
            
            loss = 0
            if logits.size(0) > 0:
                loss = criterion(logits, labels)

            # Backward pass
            loss.backward(loss)
            optimizer.step()
            
            # Accumulate for reporting
            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}/{config['epochs']}")
        print(f"Train Loss: {total_loss / len(train_dataset):.4f}")
    
    return model

def evaluate_coarse_grained(model, eval_data, k):
    dataset_contains_feasibility = "feasible" in eval_data[0].keys()
    if dataset_contains_feasibility:
        eval_set = [example for example in eval_data if example["feasible"] is not None]
    else:
        eval_set = eval_data

    eval_result = []
    sum_column_recall_at_k = 0
    sum_table_recall_at_k = 0
    sum_table_nrmc = 0
    total_valid_examples = 0

    for example in tqdm(eval_set):
        if dataset_contains_feasibility:
            goal_columns = {" ".join(column.split(".")) for column in example["columns"]}
        else:
            goal_columns = set(example["goal answer"])

        goal_tables = {col.split(" ")[0] for col in goal_columns}

        # Make relevance predictions
        predictions = predict_relevance_coarse(model, example["question"], example["schema"])
        sorted_columns = sorted(predictions.items(), reverse=True, key=lambda pair: pair[1])
        columns, relevance = zip(*sorted_columns[:k]) if len(sorted_columns) >= k else zip(*sorted_columns)
        columns, relevance = list(columns), list(relevance)

        # Column recall@k
        relevant_columns = {column for column in columns if column in goal_columns}
        column_recall_at_k = len(relevant_columns) / len(goal_columns)
        sum_column_recall_at_k += column_recall_at_k

        # Table recall@k
        relevant_tables = {column.split(" ")[0] for column in columns if column.split(" ")[0] in goal_tables}
        table_recall_at_k = len(relevant_tables) / len(goal_tables)
        sum_table_recall_at_k += table_recall_at_k

        # Table-level n-RMC
        seen_tables = set()
        table_nrmc_cutoff = len(predictions)

        for i, (col, _) in enumerate(sorted_columns):
            table = col.split(" ")[0]
            if table in goal_tables:
                seen_tables.add(table)
            if seen_tables == goal_tables:
                table_nrmc_cutoff = i + 1  # +1 for 1-based index
                break

        table_nrmc = table_nrmc_cutoff / max(len(goal_tables), 1)
        sum_table_nrmc += table_nrmc
        total_valid_examples += 1

        eval_result.append({
            "question": example["question"],
            "goal columns": list(goal_columns),
            "top k columns": columns,
            "top k relevance": relevance,
            "column recall@k": column_recall_at_k,
            "table recall@k": table_recall_at_k,
            "table-n-RMC": table_nrmc
        })

    eval_result.append({
        "Amount of questions": len(eval_set),
        "Total column recall@k": sum_column_recall_at_k / len(eval_set),
        "Total table recall@k": sum_table_recall_at_k / len(eval_set),
        "Average table-n-RMC": sum_table_nrmc / total_valid_examples,
        "K": k
    })

    return eval_result

class SchemaLinkingDatasetCoarse(torch.utils.data.Dataset):
    def __init__(self, examples):
        self.dataset_contains_feasibility = "feasible" in examples[0].keys()

        if self.dataset_contains_feasibility:
            self.examples = [example for example in examples if example["feasible"] is not None]
        else:
            self.examples = examples
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        input, repeated_schema = prepare_input(example["question"], example["schema"])
            
        # Create binary labels tensor (1 for relevant, 0 otherwise)
        num_columns = len(repeated_schema)
        labels = torch.zeros(num_columns)
        
        # Create label mapping based on "goal answer"
        if self.dataset_contains_feasibility:
            goal_columns = {" ".join(column.split(".")) for column in example["columns"]}
        else:
            goal_columns = set(example["goal answer"])

        if not self.dataset_contains_feasibility or example["feasible"] == 1:
            for col_idx, col in enumerate(repeated_schema):
                if col in goal_columns:
                    labels[col_idx] = 1.0

            if (labels == 0).all():
                raise Exception(f"No goal columns for feasible question: {example['question']}")            

        return {
            "input": input,
            "labels": labels
        }

if __name__ == "__main__":
    run_study()