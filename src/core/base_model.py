import os
from abc import abstractmethod, ABC
import re
from tqdm import tqdm
import pandas as pd
from sql_metadata import Parser
from sqlalchemy import create_engine
from dotenv import load_dotenv
from src.common.logger import get_logger
from src.core.extract_instructions import get_query_build_instruct, SchemaKind, sanitise_query
from src.core.evaluation_metrics import precision, recall, f1_score, execution_accuracy
from mschema.schema_engine import SchemaEngine

logger = get_logger(__name__)
load_dotenv()

TASK = 'text-generation'
MAX_NEW_TOKENS = 200

DB_NAME = os.getenv('DB_NAME')
DB_PATH = os.getenv('DB_PATH')
DB_PATH_NATURAL = os.getenv('DB_PATH_NATURAL')
DB_NATURAL = os.getenv('DB_NATURAL')

class PromptStrategy(ABC):
    @abstractmethod
    def get_prompt(self, schema, question) -> str:
        raise NotImplementedError("Subclasses must implement a prompt getter method.")


class NL2SQLModel(ABC):
    def __init__(self, connection, benchmark_set: list, prompt_strategy: PromptStrategy):
        """
        Init for any NL2SQL model used for benchmarking, uses transformers for all models.

        args:
        - connection: Database connection.
        - benchmark_set: Dictionary containing the benchmark dataset, format outlined in README.
        - prompt_strategy: A specialised prompt strategy for the model, should include a method get_prompt(), that builds the desired prompt.
        """
        self.benchmark = benchmark_set
        self.tokenizer = None
        self.model = None
        self.pipe = None
        self.conn = connection
        self.prompt_strategy = prompt_strategy
        self.results = {}
        self.analysis = None

    def run(self, schema_size: SchemaKind, naturalness: bool):
        logger.info(f"Started benchmarking of {self.__class__.__name__}.")
        for idx, pair in enumerate(tqdm(self.benchmark)):
            question = pair['question']
            goal = pair['golden_query']
            schema = get_query_build_instruct(schema_size, goal, naturalness)
            prompt = self.prompt_strategy.get_prompt(schema, question)
            answer = self._prune_generated_query(self._answer_single_question(prompt))
            self.results[idx] = {'question': question, 'golden_query': goal, 'golden_result': {}, 'generated_query': answer, 'generated_result': {}}
        logger.info(f"Benchmarking finished for {self.__class__.__name__}.")

    @abstractmethod
    def _answer_single_question(self):
        """
        Abstract method for answering a single question, should return a non-pruned response.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def _prune_generated_query(self, query: str):
        # Prune everything before select
        query = re.sub(r'^(.*?)SELECT', 'SELECT', query, flags=(re.IGNORECASE | re.DOTALL))
        # Prune everything after ; (the end of the query) and return
        query = re.sub(r';.*', ';', query, flags=(re.IGNORECASE | re.DOTALL))
        # Remove \n
        return query.replace("\n", "")

def _generate_mschema():
    """
    Generate schema to M-schema DDL format with additional information
    """

    db_engine = create_engine(f'sqlite:///{DB_PATH_NATURAL}')

    schema_engine = SchemaEngine(engine=db_engine, db_name=DB_NAME)
    mschema = schema_engine.mschema
    return mschema.to_mschema()

def translate_query_to_natural(query: str) -> str:
    """
    Translates an SQL from using abbreviated column and tables names to use more natural_names
    Uses the column and table natural names CSV files.

    Args:
    - query (str): SQL query in a string format
    """
    parser = Parser(sanitise_query(query))
    tables = parser.tables
    columns = parser.columns

    # Load natural column and table names:
    table_names_natural = pd.read_csv(".local/table_names_normalised.csv", header=None, names=["old_name", "new_name"])
    column_names_natural = pd.read_csv(".local/column_names_normalised.csv", header=None, names=["old_name", "new_name", "table_name"])

    table_name_mapping = dict(zip(table_names_natural['old_name'], table_names_natural['new_name']))
    column_name_mapping = { 
        (row['old_name'], row['table_name']): row['new_name']
        for _, row in column_names_natural.iterrows()
    }

    # Replace column names based on their respective tables
    for column in columns:
        if '.' in column:  # Check if column includes the table alias
            table, col_name = column.split('.')
            if (col_name, table_name_mapping[table]) in column_name_mapping:  # Check if both the column and table exist in mapping
                new_col_name = column_name_mapping[(col_name, table_name_mapping[table])]
                query = query.replace(f'.{col_name} ', f'.{new_col_name} ')
                query = query.replace(f'.{col_name},', f'.{new_col_name},')
                query = query.replace(f'.{col_name})', f'.{new_col_name})')
        else:
            col_name = column
            table = table_name_mapping[tables[0]]
            if (col_name, table) in column_name_mapping:  # Check if both the column and table exist in mapping
                new_col_name = column_name_mapping[(col_name, table)]
                query = query.replace(f' {col_name} ', f' {new_col_name} ')
                query = query.replace(f' {col_name},', f' {new_col_name},')
                query = query.replace(f' {col_name})', f' {new_col_name})')

    # Replace table names in the query
    for table in tables:
        if table in table_name_mapping:
            query = query.replace(f" {table} ", f" {table_name_mapping[table]} ")  # Ensure space for full matches
            query = query.replace(f" {table},", f" {table_name_mapping[table]},")  # Ensure space for full matches
            query = query.replace(f" {table};", f" {table_name_mapping[table]};")  # Ensure space for full matches

    return query

if __name__ == "__main__":
    _generate_mschema()
