import pytest
from src.core.base_model import NL2SQLModel, PromptStrategy
from unittest.mock import MagicMock, patch, call


@pytest.fixture
def mock_logger():
    with patch("src.core.base_model.logger") as mock_logger_instance:
        yield mock_logger_instance


class MockPromptStrategy(PromptStrategy):
    def get_prompt(self, schema, question):
        return f"MOCK PROMPT: {schema} {question}"


class MockNL2SQLModel(NL2SQLModel):
    def __init__(self, connection, benchmark_set):
        super().__init__(connection, benchmark_set, MockPromptStrategy())
        self.tokenizer = MagicMock()
        self.model = MagicMock()
        self.pipe = MagicMock(
            return_value=[{"generated_text": "sure here is the needed sql: SELECT * FROM mock_table;, is that all?"}])

    def generate_report(self):
        return {"mock_report": True}  # Fix typo from "mock_retport"


@pytest.fixture
def mock_db_conn():
    """Mocks a database connection."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.fetchall.return_value = [("mock_result",)]
    return mock_conn


@pytest.fixture
def mock_benchmark_set():
    """Mocks a small benchmark set."""
    return [{"question": "What are the users?", "golden_query": "SELECT * FROM users;"}]


@patch('src.core.extract_instructions.get_query_build_instruct', return_value="MOCK SCHEMA")
def test_nl2sql_init(mock_get_query_build_instruct, mock_db_conn, mock_benchmark_set):
    # Arrange + Act
    model = MockNL2SQLModel(mock_db_conn, mock_benchmark_set)

    # Assert
    assert model.tokenizer is not None
    assert model.model is not None
    assert model.pipe is not None
    assert model.conn == mock_db_conn
    assert isinstance(model.prompt_strategy, MockPromptStrategy)


def test_answer_single_question(mock_db_conn, mock_benchmark_set):
    # Arrange
    model = MockNL2SQLModel(mock_db_conn, mock_benchmark_set)
    q = "How many users are there"

    # Act
    res = model._answer_single_question(q)

    # Assert
    assert res == "SELECT * FROM mock_table;"


def test_prune_generated_query(mock_db_conn, mock_benchmark_set):
    # Arrange
    model = MockNL2SQLModel(mock_db_conn, mock_benchmark_set)
    raw_query = "Hello, let's generate some SQL: SELECT name FROM users WHERE age > 30; More text here..."

    # Act
    res = model._prune_generated_query(raw_query)

    # Assert
    assert res == "SELECT name FROM users WHERE age > 30;"


@pytest.mark.parametrize("schema_kind", ["columns", "tables", "full"])
@patch("src.core.base_model.get_query_build_instruct", return_value="MOCK SCHEMA")
def test_run(mock_get_query_build_instruct, mock_db_conn, mock_benchmark_set, mock_logger, schema_kind):
    # Arrange
    model = MockNL2SQLModel(mock_db_conn, mock_benchmark_set)

    with patch.object(model, "_answer_single_question", return_value="SELECT * FROM users;"):
        with patch.object(model, "_get_query_result", return_value=[("mock_result",)]):
            # Act
            model.run(schema_size=schema_kind)

    # Assert
    assert len(model.results) == len(mock_benchmark_set)
    assert model.results[0]['generated_query'] == "SELECT * FROM users;"
    assert model.results[0]['golden_result'] == [("mock_result",)]
    assert model.results[0]['generated_result'] == [("mock_result",)]

    # Ensure logger was called
    expected_calls = [
        call.info("Started benchmarking of MockNL2SQLModel."),
        call.info("Benchmarking finished for MockNL2SQLModel."),
        call.info("Running results of database for MockNL2SQLModel."),
        call.info("Executed all queries on the database for MockNL2SQLModel."),
    ]

    mock_logger.info.assert_has_calls(expected_calls, any_order=False)


@pytest.mark.parametrize(
    "gold, generated, expected",
    [
        # Generated has distinct
        (
            "SELECT name FROM names;",
            "SELECT DISTINCT name FROM names;",
            {'tables': {'gold': [], 'generated': []}, 'columns': {'gold': [], 'generated': []}, 'clauses': {}, 'distinct': {'gold': False, 'generated': True}}
        ),
        # Gold has distinct
        (
            "SELECT DISTINCT name FROM names;",
            "SELECT name FROM names;",
            {'tables': {'gold': [], 'generated': []}, 'columns': {'gold': [], 'generated': [
            ]}, 'clauses': {}, 'distinct': {'gold': True, 'generated': False}}
        ),
        # Missing column, WHERE clause, and GROUP BY
        (
            "SELECT name, age FROM users WHERE age > 18 GROUP BY age ORDER BY name",
            "SELECT name FROM users ORDER BY name",
            {'tables': {'gold': [], 'generated': []}, 'columns': {'gold': ['name', 'age'], 'generated': ['name']}, 'clauses': {'WHERE': {
                'gold': ['AGE > 18'], 'generated': []}, 'GROUPBY': {'gold': ['AGE'], 'generated': []}}, 'distinct': {'gold': False, 'generated': False}}
        ),
        # Different table
        (
            "SELECT id FROM orders",
            "SELECT id FROM transactions",
            {'tables': {'gold': ['orders'], 'generated': ['transactions']}, 'columns': {'gold': [
            ], 'generated': []}, 'clauses': {}, 'distinct': {'gold': False, 'generated': False}}
        ),
        # Identical SQL (no errors)
        (
            "SELECT id, amount FROM transactions WHERE amount > 100",
            "SELECT id, amount FROM transactions WHERE amount > 100",
            {'tables': {'gold': [], 'generated': []}, 'columns': {'gold': [], 'generated': [
            ]}, 'clauses': {}, 'distinct': {'gold': False, 'generated': False}}
        ),
        (
            "SELECT name, age FROM users WHERE age > 18 GROUP BY age ORDER BY name",
            "SELECT name, age FROM users WHERE age > 18 GROUP BY age ORDER BY age",
            {'tables': {'gold': [], 'generated': []}, 'columns': {'gold': [], 'generated': []}, 'clauses': {
                'ORDERBY': {'gold': ['NAME '], 'generated': ['AGE ']}}, 'distinct': {'gold': False, 'generated': False}}
        )
    ],
    ids=['generated extra distinct', 'generated missing distinct', 'missing column, where and group by', 'different table', 'no errors', 'different order by']
)
def test_extract_sql_mismatch(mock_db_conn, mock_benchmark_set, gold, generated, expected):
    # Arrange
    model = MockNL2SQLModel(mock_db_conn, mock_benchmark_set)

    # Act
    result = model._extract_sql_mismatches(gold, generated)

    # Assert
    assert result == expected


def test_analyse_sql():
    # Arrange
    model = MockNL2SQLModel(MagicMock(), MagicMock())
    test_gold_set = [
        "SELECT name FROM names WHERE name > 'elviz';",
        "SELECT name, age FROM names WHERE age > 18;",
        "SELECT age FROM names ORDER BY age;",
        "SELECT age, address FROM names JOIN addresses;"
    ]
    test_generated_set = [
        "SELECT name FROM names;",
        "SELECT name FROM names WHERE age > 18;",
        "SELECT age from names;",
        "SELECT age, address FROM names;"
    ]
    expected_total_errors = {
        'table_errors': 1,
        'column_errors': 1,
        'clause_errors': 3,
        'distinct_errors': 0
    }

    # Act
    result = model._analyse_sql(test_gold_set, test_generated_set)

    # Assert
    assert result['total_errors'] == expected_total_errors


def test_analyse():
    # Arrange
    model = MockNL2SQLModel(MagicMock(), MagicMock())
    model.results = {
        0: {
            'question': 'What is the capital of France?',
            'golden_query': 'SELECT capital FROM countries WHERE name="France"',
            'golden_result': [('Paris',)],
            'generated_query': 'SELECT capital FROM countries WHERE name="France"',
            'generated_result': [('Paris',)]
        },
        1: {
            'question': 'How many people live in Germany?',
            'golden_query': 'SELECT population FROM countries WHERE name="Germany"',
            'golden_result': [(83100000,)],
            'generated_query': 'SELECT population FROM countries WHERE name="Germany"',
            'generated_result': [(83100000,)]
        },
        2: {
            'question': 'List all countries in Europe.',
            'golden_query': 'SELECT name FROM countries WHERE continent="Europe"',
            'golden_result': [('France',), ('Germany',), ('Italy',)],
            'generated_query': 'SELECT name FROM countries WHERE continent="Europe"',
            'generated_result': [('France',), ('Germany',), ('Italy',)]
        },
        3: {
            'question': 'What is the GDP of Japan?',
            'golden_query': 'SELECT gdp FROM countries WHERE name="Japan"',
            'golden_result': [(5000000,)],  # Correct expected output
            # Error: wrong WHERE clause, extra column
            'generated_query': 'SELECT gdp, population FROM countries WHERE namee="Japon"',
            # Incorrect output due to extra column
            'generated_result': [(5000000, 126000000)]
        },
        4: {
            'question': 'What is the area of Canada?',
            'golden_query': 'SELECT area FROM countries WHERE name="Canada"',
            'golden_result': [(9984670,)],
            'generated_query': 'SELECT area FROM countries WHERE name="Canada"',
            'generated_result': [(9984670,)]
        }
    }

    # Act
    model.analyse()

    # Assert
    assert model.analysis is not None
