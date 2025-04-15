import torch.nn as nn
import torch
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import re
import json
from tqdm import tqdm

TRAINED_MODEL_PATH="models/EXSL/coarse_grained_schema_linker.pth"
TRAIN_SET_PATH=".local/spider_exsl_train.json"

class ExSLcModel(nn.Module):
    def __init__(self, base_model_name):
        """
        Coarse-Grained Extractive Schema Linking Model
        
        Args:
            base_model_name: Name of the pretrained decoder-only model (e.g., "deepseek-ai/deepseek-coder-6.7b")
        """
        super().__init__()
        self.base_model = AutoModel.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)
        for param in self.base_model.parameters():
            param.requires_grad = False
        hidden_size = self.base_model.config.hidden_size
        
        # Single output for relevance (binary)
        self.w_relevance = nn.Linear(hidden_size * 2, 1, dtype=torch.bfloat16)  # Only predict relevance
        
        # Special tokens for marking columns
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model
        
        Args:
            input_ids: Tokenized input sequence
            attention_mask: Attention mask for the input
            
        Returns:
            relevance_logits: Logits for relevance of each marked column
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        last_hidden_state = outputs.last_hidden_state
        
        # Find positions of « and » tokens
        batch_size = input_ids.size(0)
        relevance_logits = []
        
        for i in range(batch_size):
            input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i])

            alpha_positions = [j for j, tok in enumerate(input_tokens) if tok == "ĠÂ<<" or tok == "Â<<"]
            omega_positions = [j for j, tok in enumerate(input_tokens) if tok == "ĠÂ>>"]
            
            E_alpha = last_hidden_state[i, alpha_positions]
            E_omega = last_hidden_state[i, omega_positions]
            
            # Concatenate embeddings
            C = torch.cat([E_alpha, E_omega], dim=-1)
            
            # Predict relevance (single logit per column)
            rho = self.w_relevance(C).squeeze(-1)  # Shape: [num_columns]
            relevance_logits.append(rho)
        
        return relevance_logits
    
def train_coarse_grained(model, train_data, config):
    # Prepare datasets
    train_dataset = SchemaLinkingDatasetCoarse(train_data, model.tokenizer, config["max_length"])
    
    # Data loader
    def custom_collate_fn(batch):
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        labels = [item["labels"] for item in batch]  # we don't stack these as the number of labels per item differs (one query might use 2 columns, another 4)
    
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=custom_collate_fn)
    
    # Optimizer with paper's parameters
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(config["device"])
            attention_mask = batch["attention_mask"].to(config["device"])
            labels = [label.to(config["device"]) for label in batch["labels"]]
            
            # Forward pass
            logits_list = model(input_ids, attention_mask)
            
            losses = []

            for i, logits in enumerate(logits_list):
                if logits.size(0) > 0:
                    loss = criterion(logits, labels[i][:logits.size(0)].float())
                    losses.append(loss)

            if losses:
                batch_loss = torch.stack(losses).mean()
            else:
                continue  # skip backward if there's no valid loss

            
            # Backward pass
            batch_loss.backward()
            optimizer.step()
            
            # Accumulate for reporting
            total_loss += batch_loss.item()
        
        print(f"Epoch {epoch + 1}/{config['epochs']}")
        print(f"Train Loss: {total_loss / len(train_loader):.4f}")
    
    return model
    
class SchemaLinkingDatasetCoarse(Dataset):
    def __init__(self, examples, tokenizer, max_length=3000):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        input, repeated_schema = prepare_input(example["question"], example["schema"])
        # Tokenize the input
        encoding = self.tokenizer(
            input,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
            
        # Create binary labels tensor (1 for relevant, 0 otherwise)
        num_columns = len(repeated_schema)
        labels = torch.zeros(num_columns)
        
        # Create label mapping based on "goal answer"
        goal_columns = set(example["goal answer"])
        
        for col_idx, col in enumerate(repeated_schema):
            if col in goal_columns:
                labels[col_idx] = 1.0

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels
        }

def predict_relevance_coarse(model, question, schema, device="auto"):
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
    
    # Tokenize
    encoding = model.tokenizer(
        input_text,
        max_length=3000,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    # Move to device
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        logits_list = model(input_ids, attention_mask)
    
    # Parse results
    predictions = {}
    if logits_list:
        logits = logits_list[0]  # Single example
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
    # Configuration matching ExSL paper
    config = {
        "base_model": "deepseek-ai/deepseek-coder-6.7b-base",
        "max_length": 3000,
        "batch_size": 16,
        "learning_rate": 5e-6,
        "weight_decay": 0.0,
        "epochs": 1,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    # Initialize model
    model = ExSLcModel(config["base_model"])
    model.to(config["device"])

    # Load train data set
    with open(TRAIN_SET_PATH, "r") as train_file:
        train_set = json.load(train_file)

    # Example data format
    question = "What are all company names that have a corresponding movie directed in the year 1999?"
    schema = "\u3010DB_ID\u3011 culture_company\n\u3010Schema\u3011\n# Table: book_club\n[\n(book_club_id:INTEGER, Primary Key, Examples: [1, 2, 3]),\n(year:INTEGER, Examples: [1989, 1990]),\n(author_or_Editor:TEXT, Examples: [Michael Nava, Donald Ward, Michael Bishop]),\n(book_Title:TEXT, Examples: [Goldenboy]),\n(publisher:TEXT, Examples: [Alyson, St. Martin's Press, William Morrow]),\n(category:TEXT, Examples: [Gay M/SF, Lesb. M/SF, Gay SF/F]),\n(result:TEXT, Examples: [Won [A ], Nom, Won])\n]\n# Table: culture_company\n[\n(company_name:TEXT, Primary Key, Examples: [Cathay Pacific Culture]),\n(type:TEXT, Examples: [Corporate, Joint Venture, Subsidiary]),\n(incorporated_in:TEXT, Examples: [China, Hong Kong]),\n(group_Equity_Shareholding:REAL, Examples: [18.77, 49.0, 60.0]),\n(book_club_id:TEXT, Examples: [1, 2, 3]),\n(movie_id:TEXT, Examples: [2, 3, 4])\n]\n# Table: movie\n[\n(movie_id:INTEGER, Primary Key, Examples: [1, 2, 3]),\n(title:TEXT, Examples: [The Boondock Saints, The Big Kahuna, Storm Catcher]),\n(year:INTEGER, Examples: [1999, 2000, 2001]),\n(director:TEXT, Examples: [Troy Duffy, John Swanbeck, Anthony Hickox]),\n(budget_million:REAL, Examples: [6.0, 7.0, 5.0]),\n(gross_worldwide:INTEGER, Examples: [30471, 3728888, 40500])\n]\n\u3010Foreign keys\u3011\nculture_company.book_club_id=book_club.book_club_id\nculture_company.movie_id=movie.movie_id"
    
    # Train the model
    trained_model = train_coarse_grained(model, train_set, config)

    predictions = predict_relevance_coarse(trained_model, question, schema, config["device"])
    print("Relevance Predictions:")
    for column, prob in predictions.items():
        print(f"{column}: {'RELEVANT' if prob > 0.5 else 'IRRELEVANT'} (prob: {prob:.2f})")

    torch.save(trained_model, TRAINED_MODEL_PATH)

    loaded_model = load_schema_linker()
    new_predictions = predict_relevance_coarse(loaded_model, question, schema, config["device"])
    print("Relevance Predictions from loaded:")
    for column, prob in new_predictions.items():
        print(f"{column}: {'RELEVANT' if prob > 0.5 else 'IRRELEVANT'} (prob: {prob:.2f})")