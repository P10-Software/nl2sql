import pytest
from unittest.mock import MagicMock
from src.core.base_model import NL2SQLModel
from src.common.reporting import create_report


class MockNL2SQLModel(NL2SQLModel):
    def __init__(self):
        super().__init__(MagicMock(), MagicMock(), MagicMock())
        self.analysis = {'execution accuracy': {'total_execution_accuracy': 0.8, 'individual_execution_accuracy': {0: True, 1: True, 2: True, 3: False, 4: True}}, 'precision': {'total_precision': 0.8, 'individual_precisions': {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.0, 4: 1.0}}, 'recall': {'total_recall': 0.8, 'individual_recalls': {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.0, 4: 1.0}}, 'f1 score': {'total_f1': 0.8, 'individual_f1s': {0: 1.0, 1: 1.0, 2: 1.0, 3: 0, 4: 1.0}}, 'SQL mismatches': {'total_errors': {'table_errors': 0, 'column_errors': 1, 'clause_errors': 1, 'distinct_errors': 0}, 'individual_errors': [{'generated_sql': 'SELECT capital FROM countries WHERE name="France"', 'gold_sql': 'SELECT capital FROM countries WHERE name="France"', 'errors': {'tables': {'gold': [], 'generated': []}, 'columns': {'gold': [], 'generated': []}, 'clauses': {}, 'distinct': {'gold': False, 'generated': False}}}, {'generated_sql': 'SELECT population FROM countries WHERE name="Germany"', 'gold_sql': 'SELECT population FROM countries WHERE name="Germany"', 'errors': {
            'tables': {'gold': [], 'generated': []}, 'columns': {'gold': [], 'generated': []}, 'clauses': {}, 'distinct': {'gold': False, 'generated': False}}}, {'generated_sql': 'SELECT name FROM countries WHERE continent="Europe"', 'gold_sql': 'SELECT name FROM countries WHERE continent="Europe"', 'errors': {'tables': {'gold': [], 'generated': []}, 'columns': {'gold': [], 'generated': []}, 'clauses': {}, 'distinct': {'gold': False, 'generated': False}}}, {'generated_sql': 'SELECT gdp, population FROM countries WHERE namee="Japon"', 'gold_sql': 'SELECT gdp FROM countries WHERE name="Japan"', 'errors': {'tables': {'gold': [], 'generated': []}, 'columns': {'gold': ['gdp', 'name', 'Japan'], 'generated': ['gdp', 'population', 'namee', 'Japon']}, 'clauses': {'WHERE': {'gold': ['NAME = JAPAN '], 'generated': ['NAMEE = JAPON ']}}, 'distinct': {'gold': False, 'generated': False}}}, {'generated_sql': 'SELECT area FROM countries WHERE name="Canada"', 'gold_sql': 'SELECT area FROM countries WHERE name="Canada"', 'errors': {'tables': {'gold': [], 'generated': []}, 'columns': {'gold': [], 'generated': []}, 'clauses': {}, 'distinct': {'gold': False, 'generated': False}}}]}, 'total sql queries': 5}


class MockTwoNL2SQL(MockNL2SQLModel):
    def __init__(self):
        super().__init__()


def test_create_report():
    # Arrange
    model = MockNL2SQLModel()
    model2 = MockTwoNL2SQL()

    # Act
    create_report([model, model2])

    # Assert
    print('test')
