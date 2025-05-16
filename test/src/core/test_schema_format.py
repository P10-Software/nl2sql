from unittest.mock import patch
import textwrap
import pytest
from src.core.schema_format import schema_filtering


@pytest.fixture
def mschema():
    return textwrap.dedent("""
    【DB_ID】 trial_metadata
    【Schema】
    # Table: test_1
    [col_a:TEXT] 
    # Table: test_2
    [col_c:INT]
    # Table: test_3
    [col_c:INT]
    # Table: test_4
    [col_b:REAL]
    [col_a:TEXT]
    [col_t:TEXT]
    # Table: test_5
    [col_a:REAL]
    """)

@pytest.mark.parametrize(
    "relevant_tables, expected",
    [
        # Case: only test_2 included
        (
            ["test_2"],
            textwrap.dedent("""
            【DB_ID】 trial_metadata
            【Schema】
            # Table: test_2
            [col_c:INT]
            """)
        ),
        # Case: no tables included
        (
            [],
            textwrap.dedent("""
            【DB_ID】 trial_metadata
            【Schema】
            """)
        ),
        # Case: multiple tables and wrong tables
        (
            ["test_1", "table_test", "test_4", "No"],
            textwrap.dedent("""
            【DB_ID】 trial_metadata
            【Schema】
            # Table: test_1
            [col_a:TEXT] 
            # Table: test_4
            [col_b:REAL]
            [col_a:TEXT]
            [col_t:TEXT] 
            """)
        ),
        # Case: all tables
        (
            ["test_1", "test_2", "test_3", "test_4"],
            textwrap.dedent("""
            【DB_ID】 trial_metadata
            【Schema】
            # Table: test_1
            [col_a:TEXT] 
            # Table: test_2
            [col_c:INT]
            # Table: test_3
            [col_c:INT]
            # Table: test_4
            [col_b:REAL]
            [col_a:TEXT]
            [col_t:TEXT]
            """)
        ),
    ]
)
@patch('src.core.schema_format.get_mschema')
def test_schema_filtering(mock_get_mschema, mschema, relevant_tables, expected):
    mock_get_mschema.return_value = mschema

    result = schema_filtering(relevant_tables)

    assert result.strip() == expected.strip()
