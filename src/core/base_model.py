from abc import abstractmethod, ABC
import re
from tqdm import tqdm
import pandas as pd
from sql_metadata import Parser
from collections import Counter
from src.common.logger import get_logger
from src.core.extract_instructions import get_query_build_instruct, SchemaKind, sanitise_query
from src.core.evaluation_metrics import precision, recall, f1_score, execution_accuracy

TASK = 'text-generation'
MAX_NEW_TOKENS = 200
logger = get_logger(__name__)


class PromptStrategy(ABC):
    @abstractmethod
    def get_prompt(self, schema, question) -> str:
        raise NotImplementedError("Subclasses must implement a prompt getter method.")


class NL2SQLModel(ABC):
    def __init__(self, connection, benchmark_set: list, prompt_strategy: PromptStrategy):
        """
        Init for any NL2SQL model used for benchmarking, uses transformers for all models.

        args:
        - connection: Postgres connection string.
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

    def run(self, schema_size: SchemaKind):
        logger.info(f"Started benchmarking of {self.__class__.__name__}.")
        for idx, pair in enumerate(tqdm(self.benchmark)):
            question = pair['question']
            goal = pair['golden_query']
            schema = get_query_build_instruct(schema_size, goal, natural_names=False)
            prompt = self.prompt_strategy.get_prompt(schema, question)
            answer = self._prune_generated_query(self._answer_single_question(prompt))
            self.results[idx] = {'question': question, 'golden_query': goal, 'golden_result': {}, 'generated_query': answer, 'generated_result': {}}
        logger.info(f"Benchmarking finished for {self.__class__.__name__}.")
        logger.info(f"Running results of database for {self.__class__.__name__}.")
        for _, res in self.results.items():
            res['golden_result'] = self._get_query_result(res['golden_query'])
            res['generated_result'] = self._get_query_result(res['generated_query'])
        logger.info(f"Executed all queries on the database for {self.__class__.__name__}.")

    @abstractmethod
    def _answer_single_question(self):
        """
        Abstract method for answering a single question, should return a non-pruned response.
        """
        raise NotImplementedError("Subclasses should implement this method.")


    def translate_query_to_natural(self, query: str) -> str:
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
        column_names_natural = pd.read_csv(".local/column_names_normalised.csv", header=None, names=["old_name", "new_name"])

        table_name_mapping = dict(zip(table_names_natural['old_name'], table_names_natural['new_name']))
        column_name_mapping = dict(zip(column_names_natural['old_name'], column_names_natural['new_name']))

        for table in tables:
            if table in table_names_natural['old_name'].values:
                query = query.replace(table, table_name_mapping[table])

        for column in columns:
            if column in column_names_natural['old_name'].values:
                query = query.replace(column, column_name_mapping[column])

        return query


    def _prune_generated_query(self, query: str):
        # Prune everything before select
        query = re.sub(r'^(.*?)SELECT', 'SELECT', query, flags=(re.IGNORECASE | re.DOTALL))
        # Prune everything after ; (the end of the query) and return
        query = re.sub(r';.*', ';', query, flags=(re.IGNORECASE | re.DOTALL))
        # Remove \n
        return query.replace("\n", "")

    def _get_query_result(self, query: str):
        try:
            cur = self.conn.cursor()
            cur.execute(query)
            res = cur.fetchall()
        except Exception as e:
            logger.error(f"Error executing query: {query}\n{e}")
            res = []
        finally:
            cur.close()
        return res
