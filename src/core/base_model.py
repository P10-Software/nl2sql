from abc import abstractmethod, ABC
import re
from tqdm import tqdm
from src.common.logger import get_logger
from src.core.schema_format import get_mschema, get_DDL
from src.core.schema_chunking import chunk_mschema
from src.core.extractive_schema_linking import load_schema_linker, get_focused_schema
from src.core.abstention_classifier import load_abstention_classifier

logger = get_logger(__name__)

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

    def run(self, with_sgam):
        logger.info(f"Started benchmarking of {self.__class__.__name__}.")

        schema = get_mschema() if self.mschema else get_DDL()
        if with_sgam:
            schema_linker = load_schema_linker()
            chunks = chunk_mschema(schema, schema_linker.tokenizer, False, k=1)
            abstention_classfier = load_abstention_classifier()

        for idx, pair in enumerate(tqdm(self.benchmark)):
            question = pair['question']
            goal = pair['golden_query']

            if with_sgam:
                focused_schema = get_focused_schema(schema_linker, question, chunks, schema)
                abstention_prediction = abstention_classfier.classify(question, focused_schema)
                if abstention_prediction == "feasible":
                    answer = self._answer_single_question(question, focused_schema)
                else:
                    answer = None
            else:
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
