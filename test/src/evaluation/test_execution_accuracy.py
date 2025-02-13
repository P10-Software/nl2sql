import sqlite3
import pytest
from unittest.mock import patch
from src.evaluation.execution_accuracy import execution_accuracy


class TestExecutionAccuracy:
    @patch("src.evaluation.execution_accuracy.execute_query")
    def test_execution_accuracy_match(self, mock_execute_query):
        mock_execute_query.side_effect = [
            [("id1", "Joel"), ("id2", "SmallJoel")],
            [("id1", "Joel"), ("id2", "SmallJoel")]
        ]

        goal_query = "SELECT * FROM tab_pln"
        generated_gurey = "SELECT * FROM tab_pln"

        assert execution_accuracy(goal_query, generated_gurey)
    

    @patch("src.evaluation.execution_accuracy.execute_query")
    def test_execution_accuracy_different(self, mock_execute_query):
        mock_execute_query.side_effect = [
            [("id1", "Joel"), ("id2", "SmallJoel")],
            [("id1", "Joel"), ("id2", "SmallJoel"), ("id3", "NormalJoel")]
        ]

        goal_query = "SELECT * FROM tab_pln LIMIT 2"
        generated_gurey = "SELECT * FROM nu_pln"

        assert not execution_accuracy(goal_query, generated_gurey)

    @patch("src.evaluation.execution_accuracy.execute_query")
    def test_execution_accuracy_match_distinct(self, mock_execute_query):
        mock_execute_query.side_effect = [
            [("id1", "Joel"), ("id2", "SmallJoel"), ("id2", "SmallJoel")],
            [("id1", "Joel"), ("id2", "SmallJoel")]
        ]

        goal_query = "SELECT * FROM tab_pln"
        generated_gurey = "SELECT DISTINCT * FROM tab_pln"

        assert not execution_accuracy(goal_query, generated_gurey)


    # TODO I do not remember how this is handleds?
    @patch("src.evaluation.execution_accuracy.execute_query")
    def test_execution_accuracy_match_empty(self, mock_execute_query):
        mock_execute_query.side_effect = [
            [],
            []
        ]

        goal_query = "SELECT * FROM non_existing"
        generated_gurey = "SELECT DISTINCT * FROM unknown"

        assert execution_accuracy(goal_query, generated_gurey)
