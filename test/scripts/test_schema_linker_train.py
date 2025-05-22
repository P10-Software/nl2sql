from unittest.mock import patch, MagicMock

with patch.dict('sys.modules', {
    'torch': MagicMock(),
    'transformers': MagicMock(),
    'accelerate': MagicMock(),
    'optuna': MagicMock()
}):
    from scripts.schema_linker_train import evaluate_coarse_grained
    def test_evaluate_coarse_grained():
        # arrange
        question = "This is some mock question"
        schema = "This is some mock schema"
        goal_columns = ["users user_id", "orders amount"]
        input_data = [{"question": question, "schema": schema, "goal answer": goal_columns}]
        k = 5
        expected_result = [{"question": question, "goal columns": goal_columns, "RMC efficiency": 0.5, "top k columns": ["orders order_id", "users user_id", "users name", "orders amount", "orders user_id"], "top k relevance": [0.8, 0.7, 0.65, 0.65, 0.6], "column recall@k": 1.0, "table recall@k": 1.0}, 
                           {"Amount of questions": 1, "Average RMC efficiency": 0.5,"Total column recall@k": 1.0, "Total table recall@k": 1.0, "K": 5}]

        mock_return_value = {"users user_id": 0.7, "users name": 0.65, "users email": 0.50, "orders order_id": 0.8, "orders user_id": 0.6,"orders amount": 0.65}

        # act
        with patch("scripts.schema_linker_train.predict_relevance_coarse", return_value=mock_return_value):
            result = evaluate_coarse_grained(None, input_data, k)
        
        # assert
        assert result ==  expected_result