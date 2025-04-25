import torch
from transformers import AutoModel, AutoTokenizer
import re
import json
from tqdm import tqdm

TRAINED_MODEL_PATH="models/EXSL/xiyan_7B_coarse_grained_schema_linker_spider.pth"
TRAIN_SET_PATH=".local/SchemaLinker/spider_exsl_train.json"
EVAL_SET_PATH=".local/SchemaLinker/spider_exsl_test.json"
K=5
RESULT_FILE_PATH=f".local/SchemaLinker/Xiyan7B/spider_exsl_recall_at_{K}.json"

class ExSLcModel(torch.nn.Module):
    def __init__(self, base_model_name, freeze_base: bool = True):
        """
        Coarse-Grained Extractive Schema Linking Model
        
        Args:
            base_model_name: Name of the pretrained decoder-only model (e.g., "deepseek-ai/deepseek-coder-6.7b")
        """
        super().__init__()
        self.base_model = AutoModel.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
        hidden_size = self.base_model.config.hidden_size
        
        # Single output for relevance (binary)
        self.w_relevance = torch.nn.Linear(hidden_size * 2, 1, dtype=torch.bfloat16)  # Only predict relevance
        
        # Special tokens for marking columns
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.add_tokens(">>")
        self.base_model.resize_token_embeddings(len(self.tokenizer))
        
    def forward(self, prompt, freeze_base: bool = True):
        """
        Forward pass of the model
        
        Args:
            prompt: A formatted input containing schema, question and repeated schema
            
        Returns:
            relevance_logits: Logits for relevance of each marked column
        """

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        if freeze_base:
            with torch.no_grad():
                outputs = self.base_model(**inputs)
        else:
            outputs = self.base_model(**inputs)

        last_hidden_state = outputs.last_hidden_state
        
        # Find positions of « and » tokens        
        input_tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])  
        alpha_positions = [j for j, tok in enumerate(input_tokens) if tok == "<<"]
        omega_positions = [j for j, tok in enumerate(input_tokens) if tok == ">>"]
        
        embeddings_alpha = last_hidden_state[0, alpha_positions]
        embeddings_omega = last_hidden_state[0, omega_positions]
        
        # Concatenate embeddings
        column_embeddings = torch.cat([embeddings_alpha, embeddings_omega], dim=-1)
        
        # Predict relevance (single logit per column)
        return self.w_relevance(column_embeddings).squeeze(-1)  # Shape: [num_columns]
            
def train_coarse_grained(model, train_data, config):
    # Prepare dataset
    train_dataset = SchemaLinkingDatasetCoarse(train_data)
    
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
        
        for example in tqdm(train_dataset):
            optimizer.zero_grad()

            labels = example["labels"].to(config["device"])
            
            # Forward pass
            logits = model(example["input"], config["freeze_base"])
            
            loss = 0
            if logits.size(0) > 0:
                loss = criterion(logits, labels)
            
            if not loss:
                continue

            # Backward pass
            loss.backward()
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

def predict_relevance_coarse(model, question, schema):
    """
    Predict which schema elements are relevant to the question
    
    Args:
        model: Trained ExSLc model
        question: Natural language question
        schema: Database schema
        device: Device to run inference on
        
    Returns:
        Dictionary mapping (table, column) pairs to relevance probability
    """
    # Prepare input text
    input_text, repeated_schema = prepare_input(question, schema)
    
    # Predict
    model.eval()
    with torch.no_grad():
        logits = model(input_text)
    
    # Parse results
    predictions = {}
    probs = torch.sigmoid(logits).to(torch.float32).cpu().numpy()
    
    # Create predictions dictionary
    for col, prob in zip(repeated_schema, probs):
        predictions[col] = float(prob)
    
    return predictions 

def prepare_input(question, schema):
    """
    Prepare input text for schema linking as shown in Figure 3
    
    Args:
        question: Natural language question
        schema: Database schema in SQL CREATE TABLE format
        
    Returns:
        input_text: Formatted input text with marked columns
    """
    # Parse schema to get tables and columns
    tables_columns = parse_schema(schema)  # Implement this based on your schema format
    
    # Build the input text
    input_text = schema + "\n\nTo answer: " + question + "\n\nWe need columns:\n"
    repeated_schema = []

    # Add all columns with markers
    for table in tables_columns.keys():
        for column in tables_columns[table]:
            input_text += f"<< {table} {column} >>\n"
            repeated_schema.append(f"{table} {column}")
    return input_text, repeated_schema

def parse_schema(schema: str):
    columns_in_schema = {}
    
    # Split schema by tables
    table_sections = re.split(r"# Table: (\w+)", schema)[1:]  
    for i in range(0, len(table_sections), 2):
        table_name = table_sections[i].strip()  
        columns_section = table_sections[i + 1]
        columns_in_schema[table_name] = []

        # Extract column names
        column_matches = re.findall(r"\(\s*(\w+):", columns_section)
        for column in column_matches:
            columns_in_schema[table_name].append(column)

    return columns_in_schema

def load_schema_linker():
    return torch.load(TRAINED_MODEL_PATH, weights_only=False)

if __name__ == "__main__":
    # Train config
    config = {
        "base_model": "XGenerationLab/XiYanSQL-QwenCoder-7B-2502",
        "freeze_base": False,
        "learning_rate": 5e-6,
        "weight_decay": 0.0,
        "epochs": 2,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    # Initialize model
    model = ExSLcModel(config["base_model"], config["freeze_base"])
    model.to(config["device"])

    # Load train data set
    with open(TRAIN_SET_PATH, "r") as train_file:
        train_set = json.load(train_file)

    # Train and save model
    trained_model = train_coarse_grained(model, train_set, config)
    torch.save(trained_model, TRAINED_MODEL_PATH)

    # Load model
    loaded_model = load_schema_linker()

    # Load eval data and evaluate
    with open(EVAL_SET_PATH, "r") as eval_file:
        eval_set = json.load(eval_file)

    eval_result = evaluate_coarse_grained(loaded_model, eval_set, K)

    with open(RESULT_FILE_PATH, "w") as result_file:
        json.dump(eval_result, result_file, indent=4)

