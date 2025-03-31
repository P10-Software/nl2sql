from src.common.reporting import Reporter
import os
import pytest


@pytest.fixture
def mock_analysis():
    yield {
        'execution accuracy': {'total_execution_accuracy': 0.67, 'individual_execution_accuracy': {0: True, 1: True, 2: True, 3: False, 4: True, 5: False}},
        'precision': {'total_precision': 0.67, 'individual_precisions': {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.0, 4: 1.0, 5: 0}},
        'recall': {'total_recall': 0.67, 'individual_recalls': {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.0, 4: 1.0, 5: 0}},
        'f1 score': {'total_f1': 0.67, 'individual_f1s': {0: 1.0, 1: 1.0, 2: 1.0, 3: 0, 4: 1.0, 5: 0}},
        'SQL mismatches': {
            'total_errors': {'table_errors': 0, 'column_errors': 1, 'clause_errors': 1, 'distinct_errors': 0, 'not_query': 0, 'abstention_errors': 1},
            'individual_errors': [
                {'generated_sql': 'SELECT capital FROM countries WHERE name="France"',
                 'golden_sql': 'SELECT capital FROM countries WHERE name="France"',
                 'errors': {
                     'tables': {'golden': [], 'generated': []},
                     'columns': {'golden': [], 'generated': []},
                     'clauses': {},
                     'distinct': {'golden': False, 'generated': False},
                     'not_query': False,
                     'abstention': False}},
                {'generated_sql': 'SELECT population FROM countries WHERE name="Germany"',
                 'golden_sql': 'SELECT population FROM countries WHERE name="Germany"',
                 'errors': {
                     'tables': {'golden': [], 'generated': []},
                     'columns': {'golden': [], 'generated': []},
                     'clauses': {},
                     'distinct': {'golden': False, 'generated': False},
                     'not_query': False,
                     'abstention': False}},
                {'generated_sql': 'SELECT name FROM countries WHERE continent="Europe"',
                 'golden_sql': 'SELECT name FROM countries WHERE continent="Europe"',
                 'errors': {
                     'tables': {'golden': [], 'generated': []},
                     'columns': {'golden': [], 'generated': []},
                     'clauses': {},
                     'distinct': {'golden': False, 'generated': False},
                     'not_query': False,
                     'abstention': False}},
                {'generated_sql': 'SELECT gdp, population FROM countries WHERE namee="Japon"',
                 'golden_sql': 'SELECT gdp FROM countries WHERE name="Japan"',
                 'errors': {
                     'tables': {'golden': [], 'generated': []},
                     'columns': {'golden': ['gdp'], 'generated': ['gdp', 'population']},
                     'clauses': {'WHERE': {'golden': ['NAME = JAPAN '], 'generated': ['NAMEE = JAPON ']}},
                     'distinct': {'golden': False, 'generated': False},
                     'not_query': False,
                     'abstention': False}},
                {'generated_sql': 'SELECT area FROM countries WHERE name="Canada"',
                 'golden_sql': 'SELECT area FROM countries WHERE name="Canada"',
                 'errors': {
                     'tables': {'golden': [], 'generated': []},
                     'columns': {'golden': [], 'generated': []},
                     'clauses': {},
                     'distinct': {'golden': False, 'generated': False},
                     'not_query': False,
                     'abstention': False}},
                {'generated_sql': 'SELECT area FROM countries WHERE name="Canada"',
                 'golden_sql': None,
                 'errors': {
                     'tables': {'golden': [], 'generated': []},
                     'columns': {'golden': [], 'generated': []},
                     'clauses': {},
                     'distinct': {'golden': False, 'generated': False},
                     'not_query': False,
                     'abstention': True}}]},
        'total sql queries': 6}


@pytest.fixture
def mock_results():
    yield {
        0: {
            'question': 'What is the capital of France?',
            'golden_query': 'SELECT capital FROM countries WHERE name="France"',
            'golden_result': [('Paris',)],
            'generated_query': 'SELECT capital FROM countries WHERE name="France"',
            'generated_result': [('Paris',)]
        },
        1: {
            'question': 'How many people live in Germany?',
            'golden_query': 'SELECT population FROM countries WHERE name="Germany"',
            'golden_result': [(83100000,)],
            'generated_query': 'SELECT population FROM countries WHERE name="Germany"',
            'generated_result': [(83100000,)]
        },
        2: {
            'question': 'List all countries in Europe.',
            'golden_query': 'SELECT name FROM countries WHERE continent="Europe"',
            'golden_result': [('France',), ('Germany',), ('Italy',)],
            'generated_query': 'SELECT name FROM countries WHERE continent="Europe"',
            'generated_result': [('France',), ('Germany',), ('Italy',)]
        },
        3: {
            'question': 'What is the GDP of Japan?',
            'golden_query': 'SELECT gdp FROM countries WHERE name="Japan"',
            'golden_result': [(5000000,)],
            'generated_query': 'SELECT gdp, population FROM countries WHERE namee="Japon"',
            'generated_result': [(5000000, 126000000)]
        },
        4: {
            'question': 'What is the area of Canada?',
            'golden_query': 'SELECT area FROM countries WHERE name="Canada"',
            'golden_result': [(9984670,)],
            'generated_query': 'SELECT area FROM countries WHERE name="Canada"',
            'generated_result': [(9984670,)]
        }
    }


def test_create_report(mock_analysis):
    # Arrange
    model = Reporter()
    model.analysis = []
    model.analysis.append(("model1", "1", mock_analysis))
    file_path = ".temp"

    # Act
    model.create_report(file_path)

    # Assert
    assert os.path.exists(file_path + "/report.html")

    # Tear Down
    os.remove(file_path + "/report.html")


def test_analysis(mock_results):
    # Arrange
    model = Reporter()
    model_name = 'model1'
    run = "1"

    # Act
    model.add_result(mock_results, model_name, run)

    # Assert
    assert model.analysis is not None
    assert model.analysis[0][0] == model_name
    assert model.analysis[0][1] == run


def test_analyse_sql():
    # Arrange
    model = Reporter()
    test_gold_set = [
        "SELECT name FROM names WHERE name > 'elviz';",
        "SELECT name, age FROM names WHERE age > 18;",
        "SELECT age FROM names ORDER BY age;",
        "SELECT age, address FROM names JOIN addresses;"
    ]
    test_generated_set = [
        "SELECT name FROM names;",
        "SELECT name FROM names WHERE age > 18;",
        "SELECT age from names;",
        "SELECT age, address FROM names;"
    ]
    test_nl_questions = [
        "",
        "",
        "",
        ""
    ]
    expected_total_errors = {
        'table_errors': 1,
        'column_errors': 1,
        'clause_errors': 3,
        'distinct_errors': 0,
        'not_query_errors': 0,
        'abstention_errors': 0
    }

    # Act
    result = model._analyse_sql(test_gold_set, test_generated_set, test_nl_questions)

    # Assert
    assert result['total_errors'] == expected_total_errors


@pytest.mark.parametrize(
    "gold, generated, expected",
    [
        # Generated has distinct
        (
            "SELECT name FROM names;",
            "SELECT DISTINCT name FROM names;",
            {'tables': {'golden': [], 'generated': []}, 'columns': {'golden': [], 'generated': [
            ]}, 'clauses': {}, 'distinct': {'golden': False, 'generated': True}, 'not_query': False, 'abstention': False}
        ),
        # Gold has distinct
        (
            "SELECT DISTINCT name FROM names;",
            "SELECT name FROM names;",
            {'tables': {'golden': [], 'generated': []}, 'columns': {'golden': [], 'generated': [
            ]}, 'clauses': {}, 'distinct': {'golden': True, 'generated': False}, 'not_query': False, 'abstention': False}
        ),
        # Missing column, WHERE clause, and GROUP BY
        (
            "SELECT name, age FROM users WHERE age > 18 GROUP BY age ORDER BY name",
            "SELECT name FROM users ORDER BY name",
            {'tables': {'golden': [], 'generated': []}, 'columns': {'golden': ['name', 'age'], 'generated': ['name']}, 'clauses': {'WHERE': {
                'golden': ['AGE > 18'], 'generated': []}, 'GROUPBY': {'golden': ['AGE'], 'generated': []}}, 'distinct': {'golden': False, 'generated': False}, 'not_query': False, 'abstention': False}
        ),
        # Different table
        (
            "SELECT id FROM orders",
            "SELECT id FROM transactions",
            {'tables': {'golden': ['orders'], 'generated': ['transactions']}, 'columns': {'golden': [
            ], 'generated': []}, 'clauses': {}, 'distinct': {'golden': False, 'generated': False}, 'not_query': False, 'abstention': False}
        ),
        # Generated is not a query
        (
            "SELECT id FROM orders",
            "What are the ids of orders?",
            {'tables': {'golden': [], 'generated': []}, 'columns': {'golden': [
            ], 'generated': []}, 'clauses': {}, 'distinct': {'golden': False, 'generated': False}, 'not_query': True, 'abstention': False}
        ),
        # Abstention Mismatch
        (
            "SELECT id FROM orders",
            None,
            {'tables': {'golden': [], 'generated': []}, 'columns': {'golden': [
            ], 'generated': []}, 'clauses': {}, 'distinct': {'golden': False, 'generated': False}, 'not_query': False, 'abstention': True}
        ),
        # Identical SQL (no errors)
        (
            "SELECT id, amount FROM transactions WHERE amount > 100",
            "SELECT id, amount FROM transactions WHERE amount > 100",
            {'tables': {'golden': [], 'generated': []}, 'columns': {'golden': [], 'generated': [
            ]}, 'clauses': {}, 'distinct': {'golden': False, 'generated': False}, 'not_query': False, 'abstention': False}
        ),
        (
            "SELECT name, age FROM users WHERE age > 18 GROUP BY age ORDER BY name",
            "SELECT name, age FROM users WHERE age > 18 GROUP BY age ORDER BY age",
            {'tables': {'golden': [], 'generated': []}, 'columns': {'golden': [], 'generated': []}, 'clauses': {
                'ORDERBY': {'golden': ['NAME '], 'generated': ['AGE ']}}, 'distinct': {'golden': False, 'generated': False}, 'not_query': False, 'abstention': False}
        ),
        (
            "SELECT T1.cdw_partition_key, T1.profile_id FROM planned_profile AS T1",
            "SELECT profile_id AS unique_identifier, cdw_partition_key AS partition_key FROM planned_profile",
            {'tables': {'golden': [], 'generated': []}, 'columns': {'golden': [], 'generated': []}, 'clauses': {
                'ORDERBY': {'golden': [], 'generated': []}}, 'distinct': {'golden': False, 'generated': False}, 'not_query': False, 'abstention': False}
        )
    ],
    ids=['generated extra distinct', 'generated missing distinct',
         'missing column, where and group by', 'different table', 'generated is not a query', 'abstention mismatch', 'no errors', 'different order by', 'alias for column']
)
def test_extract_sql_mismatch(gold, generated, expected):
    # Arrange
    model = Reporter()

    # Act
    result = model._extract_sql_mismatches(gold, generated)

    # Assert
    assert set(result['tables']['golden']) == set(expected['tables']['golden'])
    assert set(result['tables']['generated']) == set(
        expected['tables']['generated'])

    assert set(result['columns']['golden']) == set(expected['columns']['golden'])
    assert set(result['columns']['generated']) == set(
        expected['columns']['generated'])

    for clause in expected['clauses']:
        assert set(result['clauses'].get(clause, {}).get('golden', [])) == set(
            expected['clauses'][clause].get('golden', []))
        assert set(result['clauses'].get(clause, {}).get('generated', [])) == set(
            expected['clauses'][clause].get('generated', []))

    assert result['distinct']['golden'] == expected['distinct']['golden']
    assert result['distinct']['generated'] == expected['distinct']['generated']
    
    assert result['not_query'] == expected['not_query']
    assert result['abstention'] == expected['abstention']
