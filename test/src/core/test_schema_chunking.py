from unittest.mock import patch, MagicMock
import textwrap


with patch.dict('sys.modules', {
    'torch': MagicMock()
}):
    from src.core.schema_chunking import chunk_mschema

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

        # print(chunks[2].strip())
        # print(expected_3)
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
