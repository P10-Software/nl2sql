from unittest.mock import patch, MagicMock
import textwrap


with patch.dict('sys.modules', {
    'torch': MagicMock()
}):
    from src.core.schema_chunking import chunk_mschema_no_relations

    def test_chunk_mschema():
        mschema = textwrap.dedent("""
        【SB_ID】 trial_metadata
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

        # Create a mock tokenizer that returns a fixed number of tokens per table
        mock_tokenizer = MagicMock(side_effect=lambda x, **kwargs: {
            "input_ids": [[0] * len(x.splitlines())]  # one token per line as a stand-in
        })

        # Mock model config
        mock_model = MagicMock()
        mock_model.model.config.max_position_embeddings = 8  # Tiny limit to force chunking
        mock_model.tokenizer = mock_tokenizer

        chunks = chunk_mschema_no_relations(mschema, mock_model)

        expected_1 = textwrap.dedent("""
        【SB_ID】 trial_metadata
        【Schema】
        # Table: test_1
        [col_a:TEXT] 
        # Table: test_2
        [col_c:INT]""")

        expected_2 = textwrap.dedent("""
        【SB_ID】 trial_metadata
        【Schema】
        # Table: test_3
        [col_c:INT]""")

        expected_3 = textwrap.dedent("""
        【SB_ID】 trial_metadata
        【Schema】
        # Table: test_4
        [col_b:REAL]
        [col_a:TEXT]
        [col_t:TEXT]""")

        assert len(chunks) == 3
        assert chunks[0].strip() == expected_1.strip()
        assert chunks[1].strip() == expected_2.strip()
        assert chunks[2].strip() == expected_3.strip()
