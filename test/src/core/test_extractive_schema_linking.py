from unittest.mock import patch, MagicMock

with patch.dict('sys.modules', {
    'torch': MagicMock(),
    'transformers': MagicMock(),
    'accelerate': MagicMock()
}):
    from src.core.extractive_schema_linking import prepare_input

    def test_extract_column_names_from_mschema():
        # arrange
        schema = """
【DB_ID】 mock_db
【Schema】
# Table: users
[
(user_id:INTEGER, Primary Key, Examples: [1, 2]),
(name:TEXT, Examples: [Alice Johnson, Bob Smith]),
(email:TEXT)
]
# Table: orders
[
(order_id:INTEGER, Primary Key, Examples: [1, 2, 3]),
(user_id:INTEGER, Examples: [1, 2]),
(amount:REAL, Examples: [99.99, 149.5, 200.0])
]
"""

        expected_columns = ["users user_id", "users name", "users email", "orders order_id", "orders user_id", "orders amount"]

        # act
        _, actual_columns = prepare_input("", schema)

        # assert
        assert expected_columns == actual_columns