import torch.nn as nn
import torch
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import re
import os

TRAINED_MODEL_PATH="models/EXSL/coarse_grained_schema_linker.pth"

class ExSLcModel(nn.Module):
    def __init__(self, base_model_name):
        """
        Coarse-Grained Extractive Schema Linking Model
        
        Args:
            base_model_name: Name of the pretrained decoder-only model (e.g., "deepseek-ai/deepseek-coder-6.7b")
        """
        super().__init__()
        self.base_model = AutoModel.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)
        hidden_size = self.base_model.config.hidden_size
        
        # Single output for relevance (binary)
        self.w_relevance = nn.Linear(hidden_size * 2, 1, dtype=torch.bfloat16)  # Only predict relevance
        
        # Special tokens for marking columns
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.add_tokens(["«", "»"])
        self.base_model.resize_token_embeddings(len(self.tokenizer))
        
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
            alpha_pos = (input_ids[i] == self.tokenizer.convert_tokens_to_ids("«")).nonzero().squeeze(-1)
            omega_pos = (input_ids[i] == self.tokenizer.convert_tokens_to_ids("»")).nonzero().squeeze(-1)
            
            E_alpha = last_hidden_state[i, alpha_pos]
            E_omega = last_hidden_state[i, omega_pos]
            
            # Concatenate embeddings
            C = torch.cat([E_alpha, E_omega], dim=-1)
            
            # Predict relevance (single logit per column)
            rho = self.w_relevance(C).squeeze(-1)  # Shape: [num_columns]
            relevance_logits.append(rho)
        
        return relevance_logits
    
def train_coarse_grained(model, train_data, config):
    # Prepare datasets
    train_dataset = SchemaLinkingDatasetCoarse(train_data, model.tokenizer, config["max_length"])
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    
    # Optimizer with paper's parameters
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop for 2 epochs
    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            batch = {k: v.to(dtype=torch.bfloat16) if v.dtype.is_floating_point else v for k, v in batch.items()}

            
            input_ids = batch["input_ids"].to(config["device"])
            attention_mask = batch["attention_mask"].to(config["device"])
            labels = batch["labels"].to(config["device"])
            
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
        
        # Tokenize the input
        encoding = self.tokenizer(
            example["input"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        # Get positions of marked columns (<< >> in your case)
        input_ids = encoding["input_ids"][0]
        alpha_pos = (input_ids == self.tokenizer.convert_tokens_to_ids("<<")).nonzero().squeeze(-1)
        
        # Create binary labels tensor (1 for relevant, 0 otherwise)
        num_columns = len(alpha_pos)
        labels = torch.zeros(num_columns)
        
        # Get all marked columns from input
        marked_columns = []
        input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        current_column = []
        collecting = False
        
        for token in input_tokens:
            if token == "<<":
                collecting = True
                current_column = []
            elif token == ">>":
                collecting = False
                marked_columns.append(" ".join(current_column).strip())
            elif collecting:
                current_column.append(token)
        
        # Create label mapping based on "goal answer"
        goal_columns = set([f"{table} {column}" for table, column in 
                          [col.split() for col in example["goal answer"]]])
        
        for col_idx, col in enumerate(marked_columns):
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
    input_text = prepare_input(question, schema)
    
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
        
        # Parse the input to get column names
        input_tokens = model.tokenizer.convert_ids_to_tokens(input_ids[0])
        column_info = []
        current_column = []
        for token in input_tokens:
            if token == "«":
                current_column = []
            elif token == "»":
                column_info.append(" ".join(current_column).strip())
                current_column = []
            elif current_column is not None:
                current_column.append(token)
        
        # Create predictions dictionary
        for col, prob in zip(column_info, probs):
            table_col = tuple(col.split())  # (table_name, column_name)
            predictions[table_col] = float(prob)
    
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
    
    # Add all columns with markers
    for table in tables_columns.keys():
        for column in tables_columns[table]:
            input_text += f"« {table} {column} »\n"
    
    return input_text

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

if __name__ == "__main__":
    # Configuration matching paper's Table 12
    config = {
        "base_model": "deepseek-ai/deepseek-coder-6.7b-base",
        "max_length": 3000,
        "batch_size": 16,
        "learning_rate": 5e-6,
        "weight_decay": 0.0,
        "epochs": 2,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    # Initialize model
    model = ExSLcModel(config["base_model"])
    model.to(config["device"])

    # Example data format
    example = {
    "input": "\u3010DB_ID\u3011 culture_company\n\u3010Schema\u3011\n# Table: movie\n[\n(movie_id:INTEGER, Primary Key, Examples: [1, 2, 3]),\n(Title:TEXT, Examples: [The Boondock Saints, The Big Kahuna, Storm Catcher]),\n(Year:INTEGER, Examples: [1999, 2000, 2001]),\n(Director:TEXT, Examples: [Troy Duffy, John Swanbeck, Anthony Hickox]),\n(Budget_million:REAL, Examples: [6.0, 7.0, 5.0]),\n(Gross_worldwide:INTEGER, Examples: [30471, 3728888, 40500])\n]\n# Table: culture_company\n[\n(Company_name:TEXT, Primary Key, Examples: [Cathay Pacific Culture]),\n(Type:TEXT, Examples: [Corporate, Joint Venture, Subsidiary]),\n(Incorporated_in:TEXT, Examples: [China, Hong Kong]),\n(Group_Equity_Shareholding:REAL, Examples: [18.77, 49.0, 60.0]),\n(book_club_id:TEXT, Examples: [1, 2, 3]),\n(movie_id:TEXT, Examples: [2, 3, 4])\n]\n# Table: book_club\n[\n(book_club_id:INTEGER, Primary Key, Examples: [1, 2, 3]),\n(Year:INTEGER, Examples: [1989, 1990]),\n(Author_or_Editor:TEXT, Examples: [Michael Nava, Donald Ward, Michael Bishop]),\n(Book_Title:TEXT, Examples: [Goldenboy]),\n(Publisher:TEXT, Examples: [Alyson, St. Martin's Press, William Morrow]),\n(Category:TEXT, Examples: [Gay M/SF, Lesb. M/SF, Gay SF/F]),\n(Result:TEXT, Examples: [Won [A ], Nom, Won])\n]\n\u3010Foreign keys\u3011\nculture_company.book_club_id=book_club.book_club_id\nculture_company.movie_id=movie.movie_id\nTo answer: What are all company names that have a corresponding movie directed in the year 1999?\nWe need columns:\n<< movie movie_id >>\n<< movie Title >>\n<< movie Year >>\n<< movie Director >>\n<< movie Budget_million >>\n<< movie Gross_worldwide >>\n<< culture_company Company_name >>\n<< culture_company Type >>\n<< culture_company Incorporated_in >>\n<< culture_company Group_Equity_Shareholding >>\n<< culture_company book_club_id >>\n<< culture_company movie_id >>\n<< book_club book_club_id >>\n<< book_club Year >>\n<< book_club Author_or_Editor >>\n<< book_club Book_Title >>\n<< book_club Publisher >>\n<< book_club Category >>\n<< book_club Result >>\n",
    "goal answer": [
        "culture_company company_name",
        "culture_company movie_id",
        "movie movie_id",
        "movie year"
    ]
}

    # Train the model
    trained_model = train_coarse_grained(model, [example], config)

    # Example inference
    question = "What are all company names that have a corresponding movie directed in the year 1999?"
    schema = """\u3010DB_ID\u3011 culture_company\n\u3010Schema\u3011\n# Table: movie\n[\n(movie_id:INTEGER, Primary Key, Examples: [1, 2, 3]),\n(Title:TEXT, Examples: [The Boondock Saints, The Big Kahuna, Storm Catcher]),\n(Year:INTEGER, Examples: [1999, 2000, 2001]),\n(Director:TEXT, Examples: [Troy Duffy, John Swanbeck, Anthony Hickox]),\n(Budget_million:REAL, Examples: [6.0, 7.0, 5.0]),\n(Gross_worldwide:INTEGER, Examples: [30471, 3728888, 40500])\n]\n# Table: culture_company\n[\n(Company_name:TEXT, Primary Key, Examples: [Cathay Pacific Culture]),\n(Type:TEXT, Examples: [Corporate, Joint Venture, Subsidiary]),\n(Incorporated_in:TEXT, Examples: [China, Hong Kong]),\n(Group_Equity_Shareholding:REAL, Examples: [18.77, 49.0, 60.0]),\n(book_club_id:TEXT, Examples: [1, 2, 3]),\n(movie_id:TEXT, Examples: [2, 3, 4])\n]\n# Table: book_club\n[\n(book_club_id:INTEGER, Primary Key, Examples: [1, 2, 3]),\n(Year:INTEGER, Examples: [1989, 1990]),\n(Author_or_Editor:TEXT, Examples: [Michael Nava, Donald Ward, Michael Bishop]),\n(Book_Title:TEXT, Examples: [Goldenboy]),\n(Publisher:TEXT, Examples: [Alyson, St. Martin's Press, William Morrow]),\n(Category:TEXT, Examples: [Gay M/SF, Lesb. M/SF, Gay SF/F]),\n(Result:TEXT, Examples: [Won [A ], Nom, Won])\n]\n\u3010Foreign keys\u3011\nculture_company.book_club_id=book_club.book_club_id\nculture_company.movie_id=movie.movie_id
    """

    predictions = predict_relevance_coarse(trained_model, question, schema, config["device"])
    print("Relevance Predictions:")
    for column, prob in predictions.items():
        print(f"{column}: {'RELEVANT' if prob > 0.5 else 'IRRELEVANT'} (prob: {prob:.2f})")

    torch.save(trained_model, TRAINED_MODEL_PATH)
