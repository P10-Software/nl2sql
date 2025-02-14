from unittest.mock import patch
import pytest
from src.core.execution_accuracy import execution_accuracy


class TestExecutionAccuracy:
    @patch("src.core.execution_accuracy.execute_query")
    def test_execution_accuracy_match(self, mock_execute_query):
        mock_execute_query.side_effect = [
            [("id1", "Joel"), ("id2", "SmallJoel")],
            [("id1", "Joel"), ("id2", "SmallJoel")]
        ]

        goal_query = "SELECT * FROM tab_pln"
        generated_gurey = "SELECT * FROM tab_pln"

        assert execution_accuracy(goal_query, generated_gurey)


    @patch("src.core.execution_accuracy.execute_query")
    def test_execution_accuracy_different(self, mock_execute_query):
        mock_execute_query.side_effect = [
            [("id1", "Joel"), ("id2", "SmallJoel")],
            [("id1", "Joel"), ("id2", "SmallJoel"), ("id3", "NormalJoel")]
        ]

        goal_query = "SELECT * FROM tab_pln LIMIT 2"
        generated_gurey = "SELECT * FROM nu_pln"

        assert not execution_accuracy(goal_query, generated_gurey)

    @patch("src.core.execution_accuracy.execute_query")
    def test_execution_accuracy_match_distinct(self, mock_execute_query):
        mock_execute_query.side_effect = [
            [("id1", "Joel"), ("id2", "SmallJoel"), ("id2", "SmallJoel")],
            [("id1", "Joel"), ("id2", "SmallJoel")]
        ]

        goal_query = "SELECT * FROM tab_pln"
        generated_gurey = "SELECT DISTINCT * FROM tab_pln"

        assert not execution_accuracy(goal_query, generated_gurey)


    @patch("src.core.execution_accuracy.execute_query")
    def test_execution_accuracy_empty_generated (self, mock_execute_query):
        mock_execute_query.side_effect = [
            [("id1", "Joel"), ("id2", "SmallJoel")],
            []
        ]

        goal_query = "SELECT DISTINCT * FROM tab_pln"
        generated_gurey = "SELECT DISTINCT * FROM unknown"

        assert not execution_accuracy(goal_query, generated_gurey)
