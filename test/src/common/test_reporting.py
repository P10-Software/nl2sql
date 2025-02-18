import pytest
from unittest.mock import MagicMock
from src.core.base_model import NL2SQLModel
from src.common.reporting import create_report


class MockNL2SQLModel(NL2SQLModel):
    def __init__(self):
        super().__init__(MagicMock(), MagicMock(), MagicMock())
        self.analysis = {}


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
