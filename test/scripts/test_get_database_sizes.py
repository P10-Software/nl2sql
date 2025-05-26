from scripts.get_database_sizes import parse_mschema

def test_parse_mscehma():
    # Arrange
    mschema = """
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
    expected = {
    'per_table': {
        'users': 3,
        'orders': 3,
        'location': 3
    },
    'total': 9
}
    
    # Act
    actual = parse_mschema(mschema)

    # Assert
    assert expected == actual
