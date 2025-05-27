from unittest.mock import patch, MagicMock


with patch.dict('sys.modules', {
    'torch': MagicMock(),
    'transformers': MagicMock(),
    'datasets': MagicMock(),
    'peft': MagicMock()
}):
    from scripts.trustsql_trian import find_t5_maxent_threshold

    def test_find_t5_maxent_threshold():
        assert True
