from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from abc import abstractmethod, ABC
from src.core.extract_instructions import get_query_build_instruct, SchemaKind
from tqdm import tqdm
from src.common.logger import get_logger
import re
from evaluation_metrics import precision, recall, f1_score

TASK = 'text-generation'
MAX_NEW_TOKENS = 200
logger = get_logger(__name__)


class PromptStrategy(ABC):
    @abstractmethod
    def get_prompt(self, schema, question) -> str:
        raise NotImplementedError("Subclasses must implement a prompt getter method.")


class NL2SQLModel(ABC):
    def __init__(self, connection, benchmark_set: dict, prompt_strategy: PromptStrategy):
        """
        Init for any NL2SQL model used for benchmarking, uses huggingface for all models.

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
            schema = get_query_build_instruct(schema_size, goal)
            prompt = self.prompt_strategy.get_prompt(schema, question)
            answer = self._answer_single_question(prompt)
            self.results[idx] = {'question': question, 'golden_query': goal, 'golden_result': {}, 'generated_query': answer, 'generated_result': {}}
        logger.info(f"Benchmarking finished for {self.__class__.__name__}.")
        logger.info(f"Running results of database for {self.__class__.__name__}.")
        for id, res in self.results.items():
            res['golden_result'] = self._get_query_result(res['golden_query'])
            res['generated_result'] = self._get_query_result(res['generated_query'])
        logger.info(f"Executed all queries on the database for {self.__class__.__name__}.")

    def analyse(self) -> None:
        """
        Generates an analysis of the results.
        Runs metrics of EX, recall, precision and F1. Also analysis SQL errors, categorising by table, column and clause.
        """
        if self.results == {}:
            logger.error("Analysis called on empty result.")
            return
        golden_results = [res['golden_result'] for res in self.results.values()]
        generated_results = [res['generated_result'] for res in self.results.values()]
        p = precision(golden_results, generated_results)
        r = recall(golden_results, generated_results)
        f1 = f1_score(golden_results, generated_results)

        pass

    @abstractmethod
    def generate_report(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def _answer_single_question(self, question: str):
        return self._prune_generated_query((self.pipe(question, return_full_text=False))[0]['generated_text'])

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
