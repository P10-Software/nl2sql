from unittest.mock import patch, MagicMock

with patch.dict('sys.modules', {
    'torch': MagicMock(),
    'transformers': MagicMock(),
    'accelerate': MagicMock()
}):
    from src.core.extractive_schema_linking import prepare_input, evaluate_coarse_grained

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

    def test_evaluate_coarse_grained():
        # arrange
        question = "This is some mock question"
        schema = "This is some mock schema"
        goal_columns = ["users user_id", "orders amount"]
        input_data = [{"question": question, "schema": schema, "goal answer": goal_columns}]
        k = 5
        expected_result = [{"question": question, "goal columns": goal_columns, "top k columns": ["orders order_id", "users user_id", "users name", "orders amount", "orders user_id"], "top k relevance": [0.8, 0.7, 0.65, 0.65, 0.6], "column recall@k": 1.0, "table recall@k": 1.0}, 
                           {"Amount of questions": 1, "Total column recall@k": 1.0, "Total table recall@k": 1.0, "K": 5}]

        mock_return_value = {"users user_id": 0.7, "users name": 0.65, "users email": 0.50, "orders order_id": 0.8, "orders user_id": 0.6,"orders amount": 0.65}

        # act
        with patch("src.core.extractive_schema_linking.predict_relevance_coarse", return_value=mock_return_value):
            result = evaluate_coarse_grained(None, input_data, k)
        
        # assert
        assert result ==  expected_result