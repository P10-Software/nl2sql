from unittest.mock import patch, MagicMock
import pytest


with patch.dict('sys.modules', {
    'torch': MagicMock(),
    'transformers': MagicMock(),
    'datasets': MagicMock(),
    'peft': MagicMock()
}):
    from scripts.trustsql_trian import find_t5_maxent_threshold

    def test_find_t5_maxent_threshold():
        input = [(1, 0.3), (0, 0.5), (1, 0.1), (0, 0.9), (1, 2), (1, 0.3), (0, 0.2), (1, 0.3), (1, 1.8)]

        threshold = find_t5_maxent_threshold(input)

        expected_threshold = 0.3

        assert pytest.approx(threshold, 0.001) == expected_threshold
