import pytest
import torch
from unittest.mock import MagicMock, patch
from src.core.abstention_classifier import AbstentionClassifier, PrefixConstrainedLogitsProcessor


@pytest.fixture
def classifier():
    with patch("src.core.abstention_classifier.AutoModelForCausalLM.from_pretrained") as mock_model_cls, \
         patch("src.core.abstention_classifier.AutoTokenizer.from_pretrained") as mock_tokenizer_cls:

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.decode = MagicMock(
            side_effect=lambda x: "answerable" if list(x) == [100, 200] else "unanswerable"
        )
        mock_tokenizer_cls.return_value = mock_tokenizer

        # Mock model with fixed device and output
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model

        clf = AbstentionClassifier()
        clf.allowed_token_sequences = [[100, 200], [101, 201]]
        clf.prefix_tree = clf._build_prefix_tree(clf.allowed_token_sequences)
        clf.logits_processor = PrefixConstrainedLogitsProcessor(clf.prefix_tree)
        return clf



def test_prompt_formatting(classifier):
    # Arrange
    question = "How many users are there?"
    schema = "CREATE TABLE users (id INT);"

    # Act
    prompt = classifier._fit_prompt(question, schema)

    # Assert
    assert question in prompt
    assert schema in prompt
    assert prompt.startswith("You are a data scientist, who has to vet questions from users.")
    assert prompt.endswith("You decide the question is: ")


@pytest.mark.parametrize("logits, expected", [
    # logits for first and second token, and expected classification result
    ([100, 200], "answerable"),    # tokens matching "answerable"
    ([101, 201], "unanswerable"),  # tokens matching "unanswerable"
])
def test_classify_with_various_logits(classifier, logits, expected):
    # Arrange
    first_logits = torch.full((1, 1, 300), -float("inf"))
    second_logits = torch.full((1, 1, 300), -float("inf"))

    first_logits[0, 0, logits[0]] = 10
    second_logits[0, 0, logits[1]] = 10

    first_output = MagicMock(logits=first_logits)
    second_output = MagicMock(logits=second_logits)

    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")
    mock_model.side_effect = [first_output, second_output]

    classifier.model = mock_model

    # Act
    result = classifier.classify("Can I list users?", "CREATE TABLE users (id INT);")

    # Assert
    assert result == expected
