from torch.nn import Module, Linear, BCEWithLogitsLoss
from torch import bfloat16, cat, stack, zeros, no_grad, float32, sigmoid, cuda, save, load
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import re
import json
from tqdm import tqdm

TRAINED_MODEL_PATH="models/EXSL/coarse_grained_schema_linker_no_batch.pth"
TRAIN_SET_PATH=".local/spider_exsl_train.json"

class ExSLcModel(Module):
    def __init__(self, base_model_name):
        """
        Coarse-Grained Extractive Schema Linking Model
        
        Args:
            base_model_name: Name of the pretrained decoder-only model (e.g., "deepseek-ai/deepseek-coder-6.7b")
        """
        super().__init__()
        self.base_model = AutoModel.from_pretrained(base_model_name, torch_dtype=bfloat16)
        for param in self.base_model.parameters():
            param.requires_grad = False
        hidden_size = self.base_model.config.hidden_size
        
        # Single output for relevance (binary)
        self.w_relevance = Linear(hidden_size * 2, 1, dtype=bfloat16)  # Only predict relevance
        
        # Special tokens for marking columns
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
    def forward(self, prompt):
        """
        Forward pass of the model
        
        Args:
            prompt: A formatted input containing schema, question and repeated schema
            
        Returns:
            relevance_logits: Logits for relevance of each marked column
        """

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        with no_grad():
            outputs = self.base_model(**inputs)

        last_hidden_state = outputs.last_hidden_state
        
        # Find positions of « and » tokens        
        input_tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])        
        alpha_positions = [j for j, tok in enumerate(input_tokens) if tok == "<<"]
        omega_positions = [j for j, tok in enumerate(input_tokens) if tok == "Ġ>>"]
        
        embeddings_alpha = last_hidden_state[0, alpha_positions]
        embeddings_omega = last_hidden_state[0, omega_positions]
        
        # Concatenate embeddings
        column_embeddings = cat([embeddings_alpha, embeddings_omega], dim=-1)
        
        # Predict relevance (single logit per column)
        return self.w_relevance(column_embeddings).squeeze(-1)  # Shape: [num_columns]
            
def train_coarse_grained(model, train_data, config):
    # Prepare datasets
    train_dataset = SchemaLinkingDatasetCoarse(train_data)
    
    # Optimizer with paper's parameters
    optimizer = AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    
    criterion = BCEWithLogitsLoss()
    
    # Training loop
    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        
        for example in tqdm(train_dataset):
            optimizer.zero_grad()

            labels = example["labels"].to(config["device"])
            
            # Forward pass
            logits = model(example["input"])
            
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
    
class SchemaLinkingDatasetCoarse(Dataset):
    def __init__(self, examples):
        self.examples = examples
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        input, repeated_schema = prepare_input(example["question"], example["schema"])
            
        # Create binary labels tensor (1 for relevant, 0 otherwise)
        num_columns = len(repeated_schema)
        labels = zeros(num_columns)
        
        # Create label mapping based on "goal answer"
        goal_columns = set(example["goal answer"])
        
        for col_idx, col in enumerate(repeated_schema):
            if col in goal_columns:
                labels[col_idx] = 1.0

        return {
            "input": input,
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
    
    # Predict
    model.eval()
    with no_grad():
        logits = model(input_text)
    
    # Parse results
    predictions = {}
    probs = sigmoid(logits).to(float32).cpu().numpy()
    
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
    return load(TRAINED_MODEL_PATH, weights_only=False)

if __name__ == "__main__":
    # Configuration matching ExSL paper
    config = {
        "base_model": "deepseek-ai/deepseek-coder-6.7b-base",
        "learning_rate": 5e-6,
        "weight_decay": 0.0,
        "epochs": 1,
        "device": "cuda" if cuda.is_available() else "cpu"
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

    save(trained_model, TRAINED_MODEL_PATH)

    loaded_model = load_schema_linker()
    new_predictions = predict_relevance_coarse(loaded_model, question, schema, config["device"])
    print("Relevance Predictions from loaded:")
    for column, prob in new_predictions.items():
        print(f"{column}: {'RELEVANT' if prob > 0.5 else 'IRRELEVANT'} (prob: {prob:.2f})")