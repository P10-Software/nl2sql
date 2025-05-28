from unittest.mock import patch, MagicMock
import pytest


with patch.dict('sys.modules', {
    'torch': MagicMock(),
    'transformers': MagicMock(),
    'datasets': MagicMock(),
    'peft': MagicMock()
}):
    from scripts.trustsql_trian import find_t5_maxent_threshold

    threshold_test_data = [
        ([(1, 0.3), (0, 0.5), (1, 0.1), (0, 0.9), (1, 2), (1, 0.3), (0, 0.2), (1, 0.3), (1, 1.8)], 0.3),
        ([(1, 0.2)], 0.2),
        ([(0, 0.2)], 0.2),
        ([(1, 0.2), (1, 0.5), (1, 1.0), (1, 1.2), (1, 2.2)], 2.2),
        ([(0, 0.2), (0, 0.5), (0, 1.0), (0, 1.2), (0, 2.2), (1, 1.5)], 0.2), # How should this be handled??
    ]

    @pytest.mark.parametrize("input, expected_threshold", threshold_test_data)
    def test_find_t5_maxent_threshold(input, expected_threshold):
        threshold = find_t5_maxent_threshold(input)

        assert pytest.approx(threshold, 0.001) == expected_threshold
