from unittest.mock import MagicMock
import textwrap
import pytest

from src.core.schema_chunking import chunk_mschema, _find_relations

def test_chunk_mschema_no_relations():
    mschema = textwrap.dedent("""
    【SB_ID】 trial_metadata
    【Schema】
    # Table: test_1
    [(col_a:TEXT)]
    # Table: test_2
    [(col_c:INT)]
    # Table: test_3
    [(col_c:INT)]
    # Table: test_4
    [(col_b:REAL),
    (col_a:TEXT),
    (col_t:TEXT)]
    """)

    # Create a mock tokenizer that returns a fixed number of tokens per table
    mock_tokenizer = MagicMock(side_effect=lambda x, **kwargs: {
        "input_ids": [[0] * len(x.splitlines())]  # one token per line as a stand-in
    })
    mock_tokenizer.model_max_length = 8

    chunks = chunk_mschema(mschema, mock_tokenizer, False)

    expected_1 = textwrap.dedent("""
    【SB_ID】 trial_metadata
    【Schema】
    # Table: test_1
    [(col_a:TEXT)]
    # Table: test_2
    [(col_c:INT)]""")

    expected_2 = textwrap.dedent("""
    【SB_ID】 trial_metadata
    【Schema】
    # Table: test_3
    [(col_c:INT)]""")

    expected_3 = textwrap.dedent("""
    【SB_ID】 trial_metadata
    【Schema】
    # Table: test_4
    [(col_b:REAL),
    (col_a:TEXT),
    (col_t:TEXT)]
    """)

    assert len(chunks) == 3

    actual_lines = set(line.strip() for line in chunks[0].strip().splitlines() if line.strip())
    expected_lines = set(line.strip() for line in expected_1.strip().splitlines() if line.strip())
    assert actual_lines == expected_lines

    actual_lines = set(line.strip() for line in chunks[1].strip().splitlines() if line.strip())
    expected_lines = set(line.strip() for line in expected_2.strip().splitlines() if line.strip())
    assert actual_lines == expected_lines

    actual_lines = set(line.strip() for line in chunks[2].strip().splitlines() if line.strip())
    expected_lines = set(line.strip() for line in expected_3.strip().splitlines() if line.strip())
    assert actual_lines == expected_lines


def test_chunk_mschema_no_relations_with_relations():
    mschema = textwrap.dedent("""
    【SB_ID】 trial_metadata
    【Schema】
    # Table: test_1
    [(col_a:TEXT)]
    # Table: test_2
    [(col_c:INT)]
    # Table: test_3
    [(col_c:INT)]
    # Table: test_4
    [(col_b:REAL),
    (col_a:TEXT),
    (col_t:TEXT)]
    【Foreign keys】
    test_4.col_a=test_1.col_a
    """)

    # Create a mock tokenizer that returns a fixed number of tokens per table
    mock_tokenizer = MagicMock(side_effect=lambda x, **kwargs: {
        "input_ids": [[0] * len(x.splitlines())]  # one token per line as a stand-in
    })
    mock_tokenizer.model_max_length = 8

    chunks = chunk_mschema(mschema, mock_tokenizer, False)

    expected_1 = textwrap.dedent("""
    【SB_ID】 trial_metadata
    【Schema】
    # Table: test_1
    [(col_a:TEXT)]
    # Table: test_2
    [(col_c:INT)]""")

    expected_2 = textwrap.dedent("""
    【SB_ID】 trial_metadata
    【Schema】
    # Table: test_3
    [(col_c:INT)]""")

    expected_3 = textwrap.dedent("""
    【SB_ID】 trial_metadata
    【Schema】
    # Table: test_4
    [(col_b:REAL),
    (col_a:TEXT),
    (col_t:TEXT)]
    """)

    assert len(chunks) == 3
    
    actual_lines = set(line.strip() for line in chunks[0].strip().splitlines() if line.strip())
    expected_lines = set(line.strip() for line in expected_1.strip().splitlines() if line.strip())
    assert actual_lines == expected_lines

    actual_lines = set(line.strip() for line in chunks[1].strip().splitlines() if line.strip())
    expected_lines = set(line.strip() for line in expected_2.strip().splitlines() if line.strip())
    assert actual_lines == expected_lines

    actual_lines = set(line.strip() for line in chunks[2].strip().splitlines() if line.strip())
    expected_lines = set(line.strip() for line in expected_3.strip().splitlines() if line.strip())
    assert actual_lines == expected_lines


def test_chunk_mschema_relations():
    mschema = textwrap.dedent("""
    【SB_ID】 trial_metadata
    【Schema】
    # Table: test_1
    [(col_a:TEXT)]
    # Table: test_2
    [(col_c:INT)]
    # Table: test_3
    [(col_c:INT),
    (col_a:TEXT)]
    # Table: test_4
    [(col_b:REAL),
    (col_a:TEXT),
    (col_t:TEXT)]
    【Foreign keys】
    test_3.col_a=test_1.col_a
    test_4.col_a=test_1.col_a""")

    # Create a mock tokenizer that returns a fixed number of tokens per table
    mock_tokenizer = MagicMock(side_effect=lambda x, **kwargs: {
        "input_ids": [[0] * len(x.splitlines())]  # one token per line as a stand-in
    })
    mock_tokenizer.model_max_length = 8

    chunks = chunk_mschema(mschema, mock_tokenizer, True)

    expected_1 = textwrap.dedent("""
    【SB_ID】 trial_metadata
    【Schema】
    # Table: test_1
    [(col_a:TEXT)]
    # Table: test_2
    [(col_c:INT)]""")

    # Expected to include the table 1 as there is a relation
    expected_2 = textwrap.dedent("""
    【SB_ID】 trial_metadata
    【Schema】
    # Table: test_3
    [(col_c:INT),
    (col_a:TEXT)]
    # Table: test_1
    [(col_a:TEXT)]
    【Foreign keys】
    test_3.col_a=test_1.col_a""")

    # Should not have space to repeat table 1 again
    expected_3 = textwrap.dedent("""
    【SB_ID】 trial_metadata
    【Schema】
    # Table: test_4
    [(col_b:REAL),
    (col_a:TEXT),
    (col_t:TEXT)]
    【Foreign keys】
    test_4.col_a=test_1.col_a""")

    assert len(chunks) == 3

    actual_lines = set(line.strip() for line in chunks[0].strip().splitlines() if line.strip())
    expected_lines = set(line.strip() for line in expected_1.strip().splitlines() if line.strip())
    assert actual_lines == expected_lines

    actual_lines = set(line.strip() for line in chunks[1].strip().splitlines() if line.strip())
    expected_lines = set(line.strip() for line in expected_2.strip().splitlines() if line.strip())
    assert actual_lines == expected_lines

    actual_lines = set(line.strip() for line in chunks[2].strip().splitlines() if line.strip())
    expected_lines = set(line.strip() for line in expected_3.strip().splitlines() if line.strip())
    assert actual_lines == expected_lines


def test_k1_chunk_mschema_relations():
    mschema = textwrap.dedent("""
    【SB_ID】 trial_metadata
    【Schema】
    # Table: test_1
    [(col_a:TEXT)]
    # Table: test_2
    [(col_c:INT)]
    # Table: test_3
    [(col_c:INT),
    (col_a:TEXT)]
    # Table: test_4
    [(col_b:REAL),
    (col_a:TEXT),
    (col_t:TEXT)]
    【Foreign keys】
    test_3.col_a=test_1.col_a
    test_4.col_a=test_1.col_a""")

    # Create a mock tokenizer that returns a fixed number of tokens per table
    mock_tokenizer = MagicMock(side_effect=lambda x, **kwargs: {
        "input_ids": [[0] * len(x.splitlines())]  # one token per line as a stand-in
    })
    mock_tokenizer.model_max_length = 8

    chunks = chunk_mschema(mschema, mock_tokenizer, True, 1)

    expected_1 = textwrap.dedent("""
    【SB_ID】 trial_metadata
    【Schema】
    # Table: test_1
    [(col_a:TEXT)]""")

    expected_2 = textwrap.dedent("""
    【SB_ID】 trial_metadata
    【Schema】
    # Table: test_2
    [(col_c:INT)]""")
    
    expected_3 = textwrap.dedent("""
    【SB_ID】 trial_metadata
    【Schema】
    # Table: test_3
    [(col_c:INT),
    (col_a:TEXT)]
    # Table: test_1
    [(col_a:TEXT)]
    【Foreign keys】
    test_3.col_a=test_1.col_a""")

    # Should not have space to repeat table 1 again
    expected_4 = textwrap.dedent("""
    【SB_ID】 trial_metadata
    【Schema】
    # Table: test_4
    [(col_b:REAL),
    (col_a:TEXT),
    (col_t:TEXT)]
    【Foreign keys】
    test_4.col_a=test_1.col_a""")

    assert len(chunks) == 4

    actual_lines = set(line.strip() for line in chunks[0].strip().splitlines() if line.strip())
    expected_lines = set(line.strip() for line in expected_1.strip().splitlines() if line.strip())
    assert actual_lines == expected_lines

    actual_lines = set(line.strip() for line in chunks[1].strip().splitlines() if line.strip())
    expected_lines = set(line.strip() for line in expected_2.strip().splitlines() if line.strip())
    assert actual_lines == expected_lines

    actual_lines = set(line.strip() for line in chunks[2].strip().splitlines() if line.strip())
    expected_lines = set(line.strip() for line in expected_3.strip().splitlines() if line.strip())
    assert actual_lines == expected_lines

    actual_lines = set(line.strip() for line in chunks[3].strip().splitlines() if line.strip())
    expected_lines = set(line.strip() for line in expected_4.strip().splitlines() if line.strip())
    assert actual_lines == expected_lines


def test_k2_chunk_mschema_relations():
    mschema = textwrap.dedent("""
    【SB_ID】 trial_metadata
    【Schema】
    # Table: test_1
    [(col_a:TEXT)]
    # Table: test_2
    [(col_c:INT)]
    # Table: test_3
    [(col_c:INT),
    (col_a:TEXT)]
    # Table: test_4
    [(col_b:REAL),
    (col_a:TEXT),
    (col_t:TEXT)]
    【Foreign keys】
    test_4.col_a=test_1.col_a""")

    # Create a mock tokenizer that returns a fixed number of tokens per table
    mock_tokenizer = MagicMock(side_effect=lambda x, **kwargs: {
        "input_ids": [[0] * len(x.splitlines())]  # one token per line as a stand-in
    })
    mock_tokenizer.model_max_length = 100

    chunks = chunk_mschema(mschema, mock_tokenizer, True, 2)

    expected_1 = textwrap.dedent("""
    【SB_ID】 trial_metadata
    【Schema】
    # Table: test_1
    [(col_a:TEXT)]
    # Table: test_2
    [(col_c:INT)]""")

    expected_2 = textwrap.dedent("""
    【SB_ID】 trial_metadata
    【Schema】
    # Table: test_3
    [(col_c:INT),
    (col_a:TEXT)]
    # Table: test_4
    [(col_b:REAL),
    (col_a:TEXT),
    (col_t:TEXT)]
    # Table: test_1
    [(col_a:TEXT)]
    【Foreign keys】
    test_4.col_a=test_1.col_a""")

    assert len(chunks) == 2
    
    actual_lines = set(line.strip() for line in chunks[0].strip().splitlines() if line.strip())
    expected_lines = set(line.strip() for line in expected_1.strip().splitlines() if line.strip())
    assert actual_lines == expected_lines

    actual_lines = set(line.strip() for line in chunks[1].strip().splitlines() if line.strip())
    expected_lines = set(line.strip() for line in expected_2.strip().splitlines() if line.strip())
    assert actual_lines == expected_lines

def test_k10_chunk_mschema_relations():
    mschema = textwrap.dedent("""
    【SB_ID】 trial_metadata
    【Schema】
    # Table: test_1
    [(col_a:TEXT)]
    # Table: test_2
    [(col_c:INT)]
    # Table: test_3
    [(col_c:INT),
    (col_a:TEXT)]
    # Table: test_4
    [(col_b:REAL),
    (col_a:TEXT),
    (col_t:TEXT)]
    【Foreign keys】
    test_3.col_a=test_1.col_a
    test_4.col_a=test_1.col_a""")

    # Create a mock tokenizer that returns a fixed number of tokens per table
    mock_tokenizer = MagicMock(side_effect=lambda x, **kwargs: {
        "input_ids": [[0] * len(x.splitlines())]  # one token per line as a stand-in
    })
    mock_tokenizer.model_max_length = 100

    chunks = chunk_mschema(mschema, mock_tokenizer, True, 10)

    expected_1 = textwrap.dedent("""
    【SB_ID】 trial_metadata
    【Schema】
    # Table: test_1
    [(col_a:TEXT)]
    # Table: test_2
    [(col_c:INT)]
    # Table: test_3
    [(col_c:INT),
    (col_a:TEXT)]
    # Table: test_4
    [(col_b:REAL),
    (col_a:TEXT),
    (col_t:TEXT)]
    【Foreign keys】
    test_3.col_a=test_1.col_a
    test_4.col_a=test_1.col_a""")

    assert len(chunks) == 1
    
    actual_lines = set(line.strip() for line in chunks[0].strip().splitlines() if line.strip())
    expected_lines = set(line.strip() for line in expected_1.strip().splitlines() if line.strip())
    assert actual_lines == expected_lines

def test_find_relations():
    mschema = textwrap.dedent("""
    【SB_ID】 trial_metadata
    【Schema】
    # Table: test_1
    [(col_a:TEXT)]
    # Table: test_2
    [(col_c:INT)]
    # Table: test_3
    [(col_c:INT),
    (col_a:TEXT)]
    # Table: test_4
    [(col_b:REAL),
    (col_a:TEXT),
    (col_c:INT)]
    【Foreign keys】
    test_3.col_a=test_1.col_a
    test_4.col_a=test_1.col_a
    test_4.col_c=test_2.col_c""")

    table = textwrap.dedent("""# Table: test_4
    [(col_b:REAL),
    (col_a:TEXT),
    (col_c:INT)]""")

    # Create a mock tokenizer that returns a fixed number of tokens per table
    mock_tokenizer = MagicMock(side_effect=lambda x, **kwargs: {
        "input_ids": [[0] * len(x.splitlines())]  # one token per line as a stand-in
    })

    # Mock model config
    mock_model = MagicMock()
    mock_model.model.config.max_position_embeddings = 8  # Tiny limit to force chunking
    mock_model.tokenizer = mock_tokenizer

    chunk_tables = set()
    chunk_relations = set()

    foreign_key_str = "【Foreign keys】"
    relations = mschema.split(foreign_key_str)[1].split()
    mschema_split = mschema.split("# ")
    mschema_tables = ['# ' + table for table in mschema_split[1:]]

    _find_relations(table, chunk_tables, chunk_relations, mschema_tables, relations, 8, mock_model)

    expected_table_relations = {'# Table: test_1\n[(col_a:TEXT)]\n', '# Table: test_2\n[(col_c:INT)]\n'}
    expected_chunk_relations = {'test_4.col_a=test_1.col_a', 'test_4.col_c=test_2.col_c'}

    assert chunk_tables == expected_table_relations
    assert chunk_relations == expected_chunk_relations


def test_k1_chunk_mschema_no_relations():
    mschema = textwrap.dedent("""
    【SB_ID】 trial_metadata
    【Schema】
    # Table: test_1
    [(col_a:TEXT)]
    # Table: test_2
    [(col_c:INT)]
    # Table: test_3
    [(col_c:INT),
    (col_a:TEXT)]
    # Table: test_4
    [(col_b:REAL),
    (col_a:TEXT),
    (col_t:TEXT)]
    【Foreign keys】
    test_3.col_a=test_1.col_a
    test_4.col_a=test_1.col_a""")

    # Create a mock tokenizer that returns a fixed number of tokens per table
    mock_tokenizer = MagicMock(side_effect=lambda x, **kwargs: {
        "input_ids": [[0] * len(x.splitlines())]  # one token per line as a stand-in
    })
    mock_tokenizer.model_max_length = 8

    chunks = chunk_mschema(mschema, mock_tokenizer, False, 1)

    expected_1 = textwrap.dedent("""
    【SB_ID】 trial_metadata
    【Schema】
    # Table: test_1
    [(col_a:TEXT)]""")

    expected_2 = textwrap.dedent("""
    【SB_ID】 trial_metadata
    【Schema】
    # Table: test_2
    [(col_c:INT)]""")
    
    expected_3 = textwrap.dedent("""
    【SB_ID】 trial_metadata
    【Schema】
    # Table: test_3
    [(col_c:INT),
    (col_a:TEXT)]""")

    # Should not have space to repeat table 1 again
    expected_4 = textwrap.dedent("""
    【SB_ID】 trial_metadata
    【Schema】
    # Table: test_4
    [(col_b:REAL),
    (col_a:TEXT),
    (col_t:TEXT)]""")

    assert len(chunks) == 4

    actual_lines = set(line.strip() for line in chunks[0].strip().splitlines() if line.strip())
    expected_lines = set(line.strip() for line in expected_1.strip().splitlines() if line.strip())
    assert actual_lines == expected_lines

    actual_lines = set(line.strip() for line in chunks[1].strip().splitlines() if line.strip())
    expected_lines = set(line.strip() for line in expected_2.strip().splitlines() if line.strip())
    assert actual_lines == expected_lines

    actual_lines = set(line.strip() for line in chunks[2].strip().splitlines() if line.strip())
    expected_lines = set(line.strip() for line in expected_3.strip().splitlines() if line.strip())
    assert actual_lines == expected_lines

    actual_lines = set(line.strip() for line in chunks[3].strip().splitlines() if line.strip())
    expected_lines = set(line.strip() for line in expected_4.strip().splitlines() if line.strip())
    assert actual_lines == expected_lines


def test_k2_chunk_mschema_no_relations():
    mschema = textwrap.dedent("""
    【SB_ID】 trial_metadata
    【Schema】
    # Table: test_1
    [(col_a:TEXT)]
    # Table: test_2
    [(col_c:INT)]
    # Table: test_3
    [(col_c:INT),
    (col_a:TEXT)]
    # Table: test_4
    [(col_b:REAL),
    (col_a:TEXT),
    (col_t:TEXT)]
    【Foreign keys】
    test_3.col_a=test_1.col_a
    test_4.col_a=test_1.col_a""")

    # Create a mock tokenizer that returns a fixed number of tokens per table
    mock_tokenizer = MagicMock(side_effect=lambda x, **kwargs: {
        "input_ids": [[0] * len(x.splitlines())]  # one token per line as a stand-in
    })
    mock_tokenizer.model_max_length = 100

    chunks = chunk_mschema(mschema, mock_tokenizer, False, 2)

    expected_1 = textwrap.dedent("""
    【SB_ID】 trial_metadata
    【Schema】
    # Table: test_1
    [(col_a:TEXT)]
    # Table: test_2
    [(col_c:INT)]""")
    
    expected_2 = textwrap.dedent("""
    【SB_ID】 trial_metadata
    【Schema】
    # Table: test_3
    [(col_c:INT),
    (col_a:TEXT)]
    # Table: test_4
    [(col_b:REAL),
    (col_a:TEXT),
    (col_t:TEXT)]""")

    assert len(chunks) == 2

    actual_lines = set(line.strip() for line in chunks[0].strip().splitlines() if line.strip())
    expected_lines = set(line.strip() for line in expected_1.strip().splitlines() if line.strip())
    assert actual_lines == expected_lines

    actual_lines = set(line.strip() for line in chunks[1].strip().splitlines() if line.strip())
    expected_lines = set(line.strip() for line in expected_2.strip().splitlines() if line.strip())
    assert actual_lines == expected_lines


def test_k10_chunk_mschema_no_relations():
    mschema = textwrap.dedent("""
    【SB_ID】 trial_metadata
    【Schema】
    # Table: test_1
    [(col_a:TEXT)]
    # Table: test_2
    [(col_c:INT)]
    # Table: test_3
    [(col_c:INT),
    (col_a:TEXT)]
    # Table: test_4
    [(col_b:REAL),
    (col_a:TEXT),
    (col_t:TEXT)]
    【Foreign keys】
    test_3.col_a=test_1.col_a
    test_4.col_a=test_1.col_a""")

    # Create a mock tokenizer that returns a fixed number of tokens per table
    mock_tokenizer = MagicMock(side_effect=lambda x, **kwargs: {
        "input_ids": [[0] * len(x.splitlines())]  # one token per line as a stand-in
    })
    mock_tokenizer.model_max_length = 100

    chunks = chunk_mschema(mschema, mock_tokenizer, False, 10)

    expected_1 = textwrap.dedent("""
    【SB_ID】 trial_metadata
    【Schema】
    # Table: test_1
    [(col_a:TEXT)]
    # Table: test_2
    [(col_c:INT)]
    # Table: test_3
    [(col_c:INT),
    (col_a:TEXT)]
    # Table: test_4
    [(col_b:REAL),
    (col_a:TEXT),
    (col_t:TEXT)]""")

    assert len(chunks) == 1

    actual_lines = set(line.strip() for line in chunks[0].strip().splitlines() if line.strip())
    expected_lines = set(line.strip() for line in expected_1.strip().splitlines() if line.strip())
    assert actual_lines == expected_lines
