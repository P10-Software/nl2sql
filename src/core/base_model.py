import os
from abc import abstractmethod, ABC
import re
from numpy import full
from tqdm import tqdm
from dotenv import load_dotenv
from src.common.logger import get_logger
from src.database.database import execute_query

logger = get_logger(__name__)
load_dotenv()

DB_NAME = os.getenv('DB_NAME')
DB_NATURAL = bool(int(os.getenv('DB_NATURAL', 0)))

class PromptStrategy(ABC):
    @abstractmethod
    def get_prompt(self, schema, question) -> str:
        raise NotImplementedError("Subclasses must implement a prompt getter method.")


class NL2SQLModel(ABC):
    def __init__(self, benchmark_set: list, prompt_strategy: PromptStrategy, mschema: bool):
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
        self.prompt_strategy = prompt_strategy
        self.results = {}
        self.analysis = None
        self.mschema = mschema

    def run(self):
        logger.info(f"Started benchmarking of {self.__class__.__name__}.")

        schema = self._get_mschema() if self.mschema else self._get_DDL()

        for idx, pair in enumerate(tqdm(self.benchmark)):
            question = pair['question']
            goal = pair['golden_query']

            answer = self._answer_single_question(question, schema)

            self.results[idx] = {
                'question': question, 
                'golden_query': goal, 
                'golden_result': {}, 
                'generated_query': answer, 
                'generated_result': {}
            }
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

    def _get_mschema(self):
        """
        Read database m-schema from file.
        """
        with open(f".local/mschema_{DB_NAME}_{'natural' if DB_NATURAL else 'abbreviated'}.txt", "r") as file:
            return file.read()

    def _get_DDL(self):
        """
        Get database DDL instructions from the database.
        """
        query = """
            SELECT sql
            FROM sqlite_master
            WHERE type IN ('table', 'index', 'view', 'trigger')
        """
        result = execute_query(query)

        ddl_statements = [row[0] for row in result]
        full_ddl = ";\n\n".join(ddl_statements) + ";"

        return full_ddl
