import torch
from transformers import AutoModel, AutoTokenizer
from src.core.schema_chunking import chunk_mschema
import re

class ExSLcModel(torch.nn.Module):
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
        self.w_relevance = torch.nn.Linear(hidden_size * 2, 1, dtype=torch.bfloat16)  # Only predict relevance
        
        # Special tokens for marking columns
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.add_tokens([">>", "<<"])
        self.base_model.resize_token_embeddings(len(self.tokenizer))
        
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

        with torch.no_grad():
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
    
def predict_relevance_for_chunks(model, question, chunks):
    predictions = list()

    for chunk in chunks:
        predictions += predict_relevance_coarse(model, question, chunk).items()
    predictions.sort(key=lambda pair: pair[1], reverse=True)
    return predictions
            
def predict_relevance_coarse(model, question, schema):
    """
    Predict which schema elements are relevant to the question
    
    Args:
        model: Trained ExSLc model
        question: Natural language question
        schema: Database schema
        
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

def load_schema_linker(model_path):
    return torch.load(model_path, weights_only=False)

def get_focused_schema(schema_linker, question, chunks, schema, threshold: int = 0.1):
    # Make relevance predictions
    predictions = predict_relevance_for_chunks(schema_linker, question, chunks)
    relevant_columns = [column for column, relevance in predictions if relevance >= threshold]
    relevant_tables_names = {column.split(" ")[0] for column in relevant_columns}

    # Remove irrelevant tables from mschema
    foreign_key_str = "【Foreign keys】"
    relations = None

    if foreign_key_str in schema:
        relations = schema.split(foreign_key_str)[1].split()
        schema = schema.split(foreign_key_str)[0]
        print(relations)

    schema_split = schema.split("# ")
    schema_header_text = schema_split[0]
    schema_tables = ['# ' + table for table in schema_split[1:] if table.split("\n")[0].split("Table: ")[1] in relevant_tables_names]

    focused_schema = schema_header_text + "".join(schema_tables)

    # Remove irrelevant relations
    if relations:
        relevant_relations = []
        for relation in relations:
            operands = relation.split("=")
            if operands[0].split(".")[0] in relevant_tables_names and operands[1].split(".")[0] in relevant_tables_names:
                relevant_relations.append(relation)

        if relevant_relations:
            focused_schema += foreign_key_str + "\n" + "".join(relevant_relations) + "\n"
    
    return focused_schema