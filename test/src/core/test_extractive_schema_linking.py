from unittest.mock import patch, MagicMock

def test_extract_column_names_from_mschema():
    with patch.dict('sys.modules', {
        'torch': MagicMock(),
        'transformers': MagicMock(),
        'accelerate': MagicMock()
    }):
        from src.core.extractive_schema_linking import prepare_input

        schema = """
【DB_ID】 mock_db
【Schema】
# Table: users
[
(user_id:INTEGER, Primary Key, Examples: [1, 2]),
(name:TEXT, Examples: [Alice Johnson, Bob Smith]),
(email:TEXT)
]
# Table: orders
[
(order_id:INTEGER, Primary Key, Examples: [1, 2, 3]),
(user_id:INTEGER, Examples: [1, 2]),
(amount:REAL, Examples: [99.99, 149.5, 200.0])
]
"""

        expected_columns = ["users user_id", "users name", "users email", "orders order_id", "orders user_id", "orders amount"]

        _, actual_columns = prepare_input("", schema)

        assert expected_columns == actual_columns

def test_get_focused_schema():
    schema = """
【DB_ID】 mock_db
【Schema】
# Table: users
[
(user_id:INTEGER, Primary Key, Examples: [1, 2]),
(name:TEXT, Examples: [Alice Johnson, Bob Smith]),
(email:TEXT)
]
# Table: orders
[
(order_id:INTEGER, Primary Key, Examples: [1, 2, 3]),
(user_id:INTEGER, Examples: [1, 2]),
(amount:REAL, Examples: [99.99, 149.5, 200.0])
]
【Foreign keys】
users.user_id=orders.order_id
"""
    expected_focused_schema = """
【DB_ID】 mock_db
【Schema】
# Table: users
[
(user_id:INTEGER, Primary Key, Examples: [1, 2]),
(name:TEXT, Examples: [Alice Johnson, Bob Smith]),
(email:TEXT)
]
"""

    mock_predictions = [
        ("users user_id", 0.1), ("users name", 0.2), ("users email", 0.05),
        ("orders order_id", 0.05), ("orders user_id", 0.09), ("orders amount", 0.01)
    ]

    with patch.dict('sys.modules', {
             'torch': MagicMock(),
             'transformers': MagicMock(),
             'accelerate': MagicMock()
         }):
        from src.core.extractive_schema_linking import get_focused_schema
        with patch('src.core.extractive_schema_linking.predict_relevance_for_chunks', return_value=mock_predictions), \
            patch('src.core.extractive_schema_linking.chunk_mschema', return_value=[schema]):
            focused_schema = get_focused_schema(None, "", [schema], schema, threshold=0.1)

    assert focused_schema == expected_focused_schema

def test_get_focused_schema_with_foreign_keys():
    schema = """
【DB_ID】 mock_db
【Schema】
# Table: users
[
(user_id:INTEGER, Primary Key, Examples: [1, 2]),
(name:TEXT, Examples: [Alice Johnson, Bob Smith]),
(email:TEXT)
]
# Table: orders
[
(order_id:INTEGER, Primary Key, Examples: [1, 2, 3]),
(user_id:INTEGER, Examples: [1, 2]),
(amount:REAL, Examples: [99.99, 149.5, 200.0])
]
# Table: location
[
(location_id:INTEGER, Primary Key, Examples: [1,2]),
(city:TEXT, Examples: [Aalborg, Copenhagen]),
(order_id:INTEGER, Example: [1,2])
]
【Foreign keys】
users.user_id=orders.order_id
orders.orders_id=location.order_id
"""
    expected_focused_schema = """
【DB_ID】 mock_db
【Schema】
# Table: users
[
(user_id:INTEGER, Primary Key, Examples: [1, 2]),
(name:TEXT, Examples: [Alice Johnson, Bob Smith]),
(email:TEXT)
]
# Table: orders
[
(order_id:INTEGER, Primary Key, Examples: [1, 2, 3]),
(user_id:INTEGER, Examples: [1, 2]),
(amount:REAL, Examples: [99.99, 149.5, 200.0])
]
【Foreign keys】
users.user_id=orders.order_id
"""

    mock_predictions = [
        ("users user_id", 0.1), ("users name", 0.2), ("users email", 0.05),
        ("orders order_id", 0.05), ("orders user_id", 0.09), ("orders amount", 0.01),
        ("location location_id", 0.02), ("location city", 0.03), ("location order_id", 0.04)
    ]

    with patch.dict('sys.modules', {
             'torch': MagicMock(),
             'transformers': MagicMock(),
             'accelerate': MagicMock()
         }):
        from src.core.extractive_schema_linking import get_focused_schema
        with patch('src.core.extractive_schema_linking.predict_relevance_for_chunks', return_value=mock_predictions), \
            patch('src.core.extractive_schema_linking.chunk_mschema', return_value=[schema]):
            focused_schema = get_focused_schema(None, "", [schema], schema, threshold=0.05)

    assert focused_schema == expected_focused_schema