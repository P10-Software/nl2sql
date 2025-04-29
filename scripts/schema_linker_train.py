import torch
import optuna
import json
from tqdm import tqdm
from src.common.logger import get_logger
from src.core.extractive_schema_linking import ExSLcModel, prepare_input, predict_relevance_coarse

TRAINED_MODEL_PATH="models/EXSL/xiyan_7B_finetuned_coarse_grained_schema_linker_spider.pth"
TRAIN_SET_PATH=".local/SchemaLinker/spider_exsl_train.json"
EVAL_SET_PATH=".local/SchemaLinker/spider_exsl_test.json"
K=5
RESULT_DIR_PATH=f".local/SchemaLinker/Xiyan7B_finetuned/"
BASE_MODEL="XGenerationLab/XiYanSQL-QwenCoder-7B-2502"
FREEZE_FINAL_LAYER=True

logger = get_logger(__name__)

def run_study():
    """
    Creates or resumes an Optuna study and launches hyperparameter
    optimization.
    """
    study = optuna.create_study(
        direction="maximize",
        study_name="schema_linker_study",
        storage="sqlite:///optuna_study_schema_linker.db",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=50)
    logger.info(f"Best trial: {study.best_trial}")

def objective(trial: optuna.trial):
    model = ExSLcModel(BASE_MODEL, FREEZE_FINAL_LAYER)

    config = {
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
        "learning_rate": trial.suggest_float("learning_rate", 5e-5, 5e-7, log=True),
        "epochs": trial.suggest_int("epochs", 2, 5)
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

    with open(RESULT_DIR_PATH + f"spider_exsl_recall_at_{K}_trial_{trial.number}.json", "w") as result_file:
        json.dump(eval_result, result_file, indent=4)
    logger.info(eval_result[-1])

    return eval_result[-1]["table recall@k"]

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
            logits = model(example["input"][0], FREEZE_FINAL_LAYER)
            
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
    # Prepare dataset
    dataset_contains_feasibility = "feasible" in eval_data[0].keys()
    if dataset_contains_feasibility:
        eval_set = [example for example in eval_data if example["feasible"] is not None]
    else:
        eval_set = eval_data

    eval_result = []
    sum_column_recall_at_k = 0
    sum_table_recall_at_k = 0
    for example in tqdm(eval_set):
        if dataset_contains_feasibility:
            goal_columns = [" ".join(column.split(".")) for column in example["columns"]]
        else:
            goal_columns = example["goal answer"]

        # Make relevance predictions
        predictions = predict_relevance_coarse(model, example["question"], example["schema"])
        columns, relevance = zip(*(sorted(predictions.items(), reverse=True, key= lambda pair: pair[1])[:k]))
        columns, relevance = list(columns), list(relevance)
    
        # Evaluate column level recall@k
        relevant_columns = [column for column in columns if column in goal_columns]
        column_recall_at_k = len(relevant_columns) / len(goal_columns)
        sum_column_recall_at_k += column_recall_at_k


        # Evaluate table level recall@k
        relevant_tables = {column.split(" ")[0] for column in relevant_columns}
        goal_tables = {column.split(" ")[0] for column in goal_columns}
        table_recall_at_k = len(relevant_tables) / len(goal_tables)
        sum_table_recall_at_k += table_recall_at_k

        eval_result.append({"question": example["question"], "goal columns": list(goal_columns), "top k columns": columns, "top k relevance": relevance, "column recall@k": column_recall_at_k, "table recall@k": table_recall_at_k})

    eval_result.append({"Amount of questions": len(eval_set), "Total column recall@k": sum_column_recall_at_k / len(eval_set), "Total table recall@k": sum_table_recall_at_k / len(eval_set), "K": k})
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