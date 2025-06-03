import pytest
from unittest.mock import MagicMock, patch, call, mock_open
import sys
sys.modules['torch'] = MagicMock()
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.utils'] = MagicMock()
sys.modules['torch.utils.data'] = MagicMock()
sys.modules['torch.optim'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['sklearn'] = MagicMock()
sys.modules['sklearn.metrics'] = MagicMock()
from src.core.base_model import NL2SQLModel, PromptStrategy


@pytest.fixture
def mock_logger():
    with patch("src.core.base_model.logger") as mock_logger_instance:
        yield mock_logger_instance


class MockPromptStrategy(PromptStrategy):
    def get_prompt(self, schema, question):
        return f"MOCK PROMPT: {schema} {question}"


class MockNL2SQLModel(NL2SQLModel):
    def __init__(self, benchmark_set):
        super().__init__(benchmark_set, MockPromptStrategy(), False)
        self.tokenizer = MagicMock()
        self.model = MagicMock()
        self.pipe = MagicMock(
            return_value=[{"generated_text": "sure here is the needed sql: SELECT * FROM mock_table;, is that all?"}])

    def generate_report(self):
        return {"mock_report": True}  # Fix typo from "mock_retport"

    def _answer_single_question(self, question):
        return self._prune_generated_query((self.pipe(question, return_full_text=False))[0]['generated_text'])

@pytest.fixture
def mock_benchmark_set():
    """Mocks a small benchmark set."""
    return [{"question": "What are the users?", "golden_query": "SELECT * FROM users;"}]


def test_nl2sql_init(mock_benchmark_set):
    # Arrange + Act
    model = MockNL2SQLModel(mock_benchmark_set)

    # Assert
    assert model.tokenizer is not None
    assert model.model is not None
    assert model.pipe is not None
    assert isinstance(model.prompt_strategy, MockPromptStrategy)


def test_answer_single_question(mock_benchmark_set):
    # Arrange
    model = MockNL2SQLModel(mock_benchmark_set)
    q = "How many users are there"

    # Act
    res = model._answer_single_question(q)

    # Assert
    assert res == "SELECT * FROM mock_table;"


def test_prune_generated_query(mock_benchmark_set):
    # Arrange
    model = MockNL2SQLModel(mock_benchmark_set)
    raw_query = "Hello, let's generate some SQL: SELECT name FROM users WHERE age > 30; More text here..."

    # Act
    res = model._prune_generated_query(raw_query)

    # Assert
    assert res == "SELECT name FROM users WHERE age > 30;"


def test_run(mock_benchmark_set, mock_logger):
    # Arrange
    model = MockNL2SQLModel(mock_benchmark_set)

    with patch.object(model, "_answer_single_question", return_value="SELECT * FROM users;"), \
         patch("builtins.open", mock_open(read_data="")):
        model.run(False)

    # Assert
    assert len(model.results) == len(mock_benchmark_set)
    assert model.results[0]['generated_query'] == "SELECT * FROM users;"
    assert model.results[0]['golden_result'] == {}
    assert model.results[0]['generated_result'] == {}

    # Ensure logger was called
    expected_calls = [
        call.info("Started benchmarking of MockNL2SQLModel."),
        call.info("Benchmarking finished for MockNL2SQLModel."),
    ]

    mock_logger.info.assert_has_calls(expected_calls, any_order=False)
