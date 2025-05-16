from unittest.mock import patch, MagicMock
import textwrap
import pytest

with patch.dict('sys.modules', {
    'torch': MagicMock()
}):
    from src.core.schema_chunking import chunk_mschema, _find_relations, mschema_to_k_chunks

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

        # Mock model config
        mock_model = MagicMock()
        mock_model.model.config.max_position_embeddings = 8  # Tiny limit to force chunking
        mock_model.tokenizer = mock_tokenizer

        chunks = chunk_mschema(mschema, mock_model, False)

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
        assert chunks[0].strip() == expected_1.strip()
        assert chunks[1].strip() == expected_2.strip()
        assert chunks[2].strip() == expected_3.strip()


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

        # Mock model config
        mock_model = MagicMock()
        mock_model.model.config.max_position_embeddings = 8  # Tiny limit to force chunking
        mock_model.tokenizer = mock_tokenizer

        chunks = chunk_mschema(mschema, mock_model, False)

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
        assert chunks[0].strip() == expected_1.strip()
        assert chunks[1].strip() == expected_2.strip()
        assert chunks[2].strip() == expected_3.strip()


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

        # Mock model config
        mock_model = MagicMock()
        mock_model.model.config.max_position_embeddings = 8  # Tiny limit to force chunking
        mock_model.tokenizer = mock_tokenizer

        chunks = chunk_mschema(mschema, mock_model, True)

        expected_1 = textwrap.dedent("""
        【SB_ID】 trial_metadata
        【Schema】
        # Table: test_1
        [(col_a:TEXT)]
        # Table: test_2
        [(col_c:INT)]
        【Foreign keys】""")

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

def test_chunks_returned_correctly():
    tokenizer = MagicMock()
    tokenizer.side_effect = lambda text, **kwargs: {
        "input_ids": [[0] * (len(text) // 5)]
    }

    mschema = "HEADER#table1#table2#table3#table4"
    chunks = mschema_to_k_chunks(mschema, tokenizer, context_size=150, k=2)

    assert len(chunks) == 2
    assert chunks[0] == "HEADER#table1#table2"
    assert len(tokenizer(chunks[0])["input_ids"][0]) <= 100 

    assert chunks[1] == "HEADER#table3#table4"
    assert len(tokenizer(chunks[1])["input_ids"][0]) <= 100 

def test_k_greater_than_table_count():
    tokenizer = MagicMock()
    tokenizer.return_value = {"input_ids": [[0] * 10]}

    mschema = "HEADER#table1#table2"
    chunks = mschema_to_k_chunks(mschema, tokenizer, context_size=100, k=10)

    assert len(chunks) == 2

def test_raises_exception_when_chunk_too_large():
    tokenizer = MagicMock()
    tokenizer.side_effect = lambda text, **kwargs: {
        "input_ids": [[0] * 200]
    }

    mschema = "HEADER#table1#table2#table3"

    with pytest.raises(Exception, match="Chunk does not fit into model"):
        mschema_to_k_chunks(mschema, tokenizer, context_size=100, k=2)