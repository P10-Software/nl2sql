from abc import abstractmethod, ABC
from src.core.extract_instructions import get_query_build_instruct, SchemaKind
from tqdm import tqdm
from src.common.logger import get_logger
import re
from src.core.evaluation_metrics import precision, recall, f1_score, execution_accuracy
import sql_metadata
from collections import Counter

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

        golden_sql = [res['golden_query'] for res in self.results.values()]
        generated_sql = [res['generated_query'] for res in self.results.values()]

        sql_errors = self._analyse_sql(golden_sql, generated_sql)

        self.analysis = {
            'execution accuracy': execution_accuracy(golden_results, generated_results),
            'precision': precision(golden_results, generated_results),
            'recall': recall(golden_results, generated_results),
            'f1 score': f1_score(golden_results, generated_results),
            'SQL mismatches': sql_errors,
            'total sql queries': len(generated_sql)
        }

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

    def _analyse_sql(self, gold_sql_list, generated_sql_list):
        """
        Finds and aggregated the error clauses for all SQL, comparing two lists of generated and gold sql queries.
        """
        error_counts = Counter({
            "table_errors": 0,
            "column_errors": 0,
            "clause_errors": 0,
            "distinct_errors": 0
        })
        individual_errors = []

        for gold_sql, generated_sql in zip(gold_sql_list, generated_sql_list):
            mismatches = self._extract_sql_mismatches(gold_sql, generated_sql)

            # Count table mismatches
            if mismatches['tables']['gold'] or mismatches['tables']['generated']:
                error_counts["table_errors"] += 1

            # Count column mismatches
            if mismatches['columns']['gold'] or mismatches['columns']['generated']:
                error_counts["column_errors"] += 1

            # Count clause mismatches
            clause_mismatch_count = sum(1 for _ in mismatches["clauses"])
            error_counts["clause_errors"] += clause_mismatch_count

            if mismatches['distinct']['gold'] != mismatches['distinct']['generated']:
                error_counts['distinct_errors'] += 1

            individual_errors.append({
                'generated_sql': generated_sql,
                'gold_sql': gold_sql,
                'errors': mismatches
            })

        return {
            'total_errors': dict(error_counts),
            'individual_errors': individual_errors
        }

    def _extract_sql_mismatches(self, gold_sql, generated_sql):
        gold_parser = sql_metadata.Parser(gold_sql)
        generated_parser = sql_metadata.Parser(generated_sql)

        mismatches = {
            'tables': {'gold': [], 'generated': []},
            'columns': {'gold': [], 'generated': []},
            'clauses': {},
            'distinct': {'gold': self._has_dictinct(gold_parser), 'generated': self._has_dictinct(generated_parser)}
        }

        gold_tables = set(gold_parser.tables) if gold_parser.tables else set()
        generated_tables = set(
            generated_parser.tables) if generated_parser.tables else set()
        if gold_tables != generated_tables:
            mismatches['tables']['gold'] = gold_parser.tables
            mismatches['tables']['generated'] = generated_parser.tables

        gold_columns = set(self._extract_columns(gold_parser))
        generated_columns = set(self._extract_columns(generated_parser))
        if gold_columns != generated_columns:
            mismatches['columns']['gold'] = list(gold_columns)
            mismatches['columns']['generated'] = list(generated_columns)

        gold_clauses = self._extract_clauses(gold_parser)
        generated_clauses = self._extract_clauses(generated_parser)

        clause_errors = {}

        for clause in gold_clauses:
            gold_values = gold_clauses[clause]
            generated_values = generated_clauses.get(clause, [])

            if gold_values != generated_values:
                clause_errors[clause] = {
                    "gold": gold_values,
                    "generated": generated_values,
                }

        mismatches['clauses'] = clause_errors

        return mismatches

    def _extract_clauses(self, parser):
        clauses = {'WHERE': [], 'JOIN': [], 'GROUPBY': [], 'ORDERBY': []}

        for token in parser.tokens:
            if token.is_keyword and token.normalized in ['WHERE', 'JOIN', 'GROUPBY', 'ORDERBY']:
                curr_tok = token.next_token
                clause_filter = []
                while curr_tok and not curr_tok.is_keyword:
                    clause_filter.append(curr_tok.normalized)
                    curr_tok = curr_tok.next_token

                clauses[token.normalized].append(" ".join(clause_filter))

        return clauses

    def _has_dictinct(self, parser):
        for token in parser.tokens:
            if token.is_keyword and token.normalized == 'DISTINCT':
                return True
        return False

    def _extract_columns(self, parser):
        columns = []
        for token in parser.tokens:
            if token.is_keyword and token.normalized in ['SELECT', 'DISTINCT']:
                next = token.next_token
                while next and not next.is_keyword:
                    if next.normalized != ',':
                        columns.append(next.normalized.lower())
                    next = next.next_token
        return columns
