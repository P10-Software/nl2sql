import pytest
from src.core.evaluation_metrics import precision, recall, f1_score, execution_accuracy


@pytest.mark.parametrize("golden, generated, result", [
    # 1. Perfect Match
    (
        [([(1, 'Alice'), (2, 'Bob')], ['name'])],
        [([(1, 'Alice'), (2, 'Bob')], ['name'])],
        {'total_precision': 1.0, 'individual_precisions': {0: 1.0}}
    ),
    # 2. Some Correct, Some Extra
    (
        [([(1, 'Alice'), (2, 'Bob')], ['name'])],
        [([(1, 'Alice'), (3, 'Bravo')], ['name'])],
        {'total_precision': 0.5, 'individual_precisions': {0: 0.5}}
    ),
    # 3. No Overlap
    (
        [([(1, 'Alice'), (2, 'Bob')], ['name'])],
        [([(3, 'Bravo'), (4, 'Delta')], ['name'])],
        {'total_precision': 0.0, 'individual_precisions': {0: 0.0}}
    ),
    # 4. Generated Set is Empty
    (
        [([(1, 'Alice'), (2, 'Bob')], ['name'])],
        [([], ['name'])],
        {'total_precision': 0.0, 'individual_precisions': {0: 0.0}}
    ),
    # 5. Gold Set is Empty
    (
        [([], ['name'])],
        [([(1, 'Alice'), (2, 'Bob')], ['name'])],
        {'total_precision': 0.0, 'individual_precisions': {0: 0.0}}
    ),
    # 6. Some Missing, Some Extra
    (
        [([(1, 'Alice'), (2, 'Bob'), (3, 'Charlie')], ['name'])],
        [([(1, 'Alice'), (4, 'Delta')], ['name'])],
        {'total_precision': 0.5, 'individual_precisions': {0: 0.5}}
    ),
    # 7. Duplicate Predictions
    (
        [([(1, 'Alice'), (2, 'Bob')], ['name'])],
        [([(1, 'Alice'), (1, 'Alice'), (2, 'Bob')], ['name'])],
        {'total_precision': 0.67, 'individual_precisions': {0: 0.67}}
    ),
    # 8. Duplicate columns
    (
        [([(1, 'Alice'), (2, 'Bob')], ['people.name'])],
        [([(1, 'Alice', 'Alice'), (2, 'Bob', 'Bob')], ['people.name', 'people.name'])],
        {'total_precision': 1.0, 'individual_precisions': {0: 1.0}}
    ),
    # 9. Different columns
    (
        [([(1, 'Alice'), (2, 'Bob')], ['people.name'])],
        [([(1, 'Alice', 'Alice'), (2, 'Bob', 'Bob')], ['people.name', 'admins.name'])],
        {'total_precision': 0.5, 'individual_precisions': {0: 0.5}}
    )
], ids=[
    'full match', 'generated one diff', 'generated full diff', 'empty generated', 'empty gold',
    'some missing some extra', 'duplicate predictions', 'duplicate columns', 'different columns'
])
def test_precision(golden, generated, result):
    # Arrange + Act
    res = precision(golden, generated)

    # Assert
    assert res == result


@pytest.mark.parametrize("golden, generated, result", [
    # 1. Perfect Match
    (
        [[(1, 'Alice'), (2, 'Bob')]],
        [[(1, 'Alice'), (2, 'Bob')]],
        {'total_recall': 1, 'individual_recalls': {0: 1.0}}
    ),
    # 2. Some Correct, Some Extra
    (
        [[(1, 'Alice'), (2, 'Bob')]],
        [[(1, 'Alice'), (3, 'Bravo')]],
        {'total_recall': 0.5, 'individual_recalls': {0: 0.5}}
    ),
    # 3. No Overlap
    (
        [[(1, 'Alice'), (2, 'Bob')]],
        [[(3, 'Bravo'), (4, 'Delta')]],
        {'total_recall': 0, 'individual_recalls': {0: 0.0}}
    ),
    # 4. Generated Set is Empty
    (
        [[(1, 'Alice'), (2, 'Bob')]],
        [[]],
        {'total_recall': 0, 'individual_recalls': {0: 0.0}}
    ),
    # 5. Gold Set is Empty
    (
        [[]],
        [[(1, 'Alice'), (2, 'Bob')]],
        {'total_recall': 0.0, 'individual_recalls': {0: 0.0}}
    ),
    # 6. Some Missing, Some Extra
    (
        [[(1, 'Alice'), (2, 'Bob'), (3, 'Charlie')]],
        [[(1, 'Alice'), (4, 'Delta')]],
        {'total_recall': 1/3, 'individual_recalls': {0: 1/3}}
    ),
    # 7. Duplicate Predictions (Should not impact recall)
    (
        [[(1, 'Alice'), (2, 'Bob')]],
        [[(1, 'Alice'), (1, 'Alice'), (2, 'Bob')]],
        {'total_recall': 1.0, 'individual_recalls': {0: 1.0}}
    )
], ids=[
    'full match', 'generated one diff', 'generated full diff', 'empty generated', 'empty gold',
    'some missing some extra', 'duplicate predictions'
])
def test_recall(golden, generated, result):
    # Arrange + Act
    res = recall(golden, generated)

    # Assert
    assert res == result


@pytest.mark.parametrize("golden, generated, result", [
    # 1. Perfect Match (Precision = 1, Recall = 1, F1 = 1)
    (
        [[(1, 'Alice'), (2, 'Bob')]],
        [[(1, 'Alice'), (2, 'Bob')]],
        {'total_f1': 1, 'individual_f1s': {0: 1.0}}
    ),
    # 2. Some Correct, Some Extra (Precision = 0.5, Recall = 0.5, F1 = 0.5)
    (
        [[(1, 'Alice'), (2, 'Bob')]],
        [[(1, 'Alice'), (3, 'Bravo')]],
        {'total_f1': 0.5, 'individual_f1s': {0: 0.5}}
    ),
    # 3. No Overlap (Precision = 0, Recall = 0, F1 = 0)
    (
        [[(1, 'Alice'), (2, 'Bob')]],
        [[(3, 'Bravo'), (4, 'Delta')]],
        {'total_f1': 0, 'individual_f1s': {0: 0.0}}
    ),
    # 4. Generated Set is Empty (Precision = 0, Recall = 0, F1 = 0)
    (
        [[(1, 'Alice'), (2, 'Bob')]],
        [[]],
        {'total_f1': 0, 'individual_f1s': {0: 0.0}}
    ),
    # 5. Gold Set is Empty (Precision = 0, Recall = 0, F1 = 0)
    (
        [[]],
        [[(1, 'Alice'), (2, 'Bob')]],
        {'total_f1': 0.0, 'individual_f1s': {0: 0.0}}
    ),
    # 6. Some Missing, Some Extra (Precision = 0.5, Recall ≈ 0.33, F1 ≈ 0.4)
    (
        [[(1, 'Alice'), (2, 'Bob'), (3, 'Charlie')]],
        [[(1, 'Alice'), (4, 'Delta')]],
        {'total_f1': 2 * (0.5 * (1/3)) / (0.5 + (1/3)),
         'individual_f1s': {0: 2 * (0.5 * (1/3)) / (0.5 + (1/3))}}
    ),
    # 7. Duplicate Predictions (Shouldn't affect F1, same as perfect match)
    (
        [[(1, 'Alice'), (2, 'Bob')]],
        [[(1, 'Alice'), (1, 'Alice'), (2, 'Bob')]],
        {'total_f1': 1.0, 'individual_f1s': {0: 1.0}}
    ),
], ids=[
    'full match', 'generated one diff', 'generated full diff', 'empty generated', 'empty gold',
    'some missing some extra', 'duplicate predictions'
])
def test_f1_score(golden, generated, result):
    # Arrange + Act
    res = f1_score(golden, generated)

    # Assert
    assert res == result


@pytest.mark.parametrize("golden, generated, expected_result", [
    # 1. Perfect Match (Execution Accuracy = 1.0)
    (
        [[(1, 'Alice'), (2, 'Bob')]],
        [[(1, 'Alice'), (2, 'Bob')]],
        {'total_execution_accuracy': 1.0, 'individual_execution_accuracy': {0: 1.0}}
    ),
    # 2. Some Correct, Some Extra (Execution Accuracy = 0.0)
    (
        [[(1, 'Alice'), (2, 'Bob')]],
        [[(1, 'Alice'), (3, 'Bravo')]],
        {'total_execution_accuracy': 0.0, 'individual_execution_accuracy': {0: 0.0}}
    ),
    # 3. No Overlap (Execution Accuracy = 0.0)
    (
        [[(1, 'Alice'), (2, 'Bob')]],
        [[(3, 'Bravo'), (4, 'Delta')]],
        {'total_execution_accuracy': 0.0, 'individual_execution_accuracy': {0: 0.0}}
    ),
    # 4. Generated Set is Empty (Execution Accuracy = 0.0)
    (
        [[(1, 'Alice'), (2, 'Bob')]],
        [[]],
        {'total_execution_accuracy': 0.0, 'individual_execution_accuracy': {0: 0.0}}
    ),
    # 5. Gold Set is Empty (Execution Accuracy = 0.0)
    (
        [[]],
        [[(1, 'Alice'), (2, 'Bob')]],
        {'total_execution_accuracy': 0.0, 'individual_execution_accuracy': {0: 0.0}}
    ),
    # 6. Some Missing, Some Extra (Execution Accuracy = 0.0)
    (
        [[(1, 'Alice'), (2, 'Bob'), (3, 'Charlie')]],
        [[(1, 'Alice'), (4, 'Delta')]],
        {'total_execution_accuracy': 0.0, 'individual_execution_accuracy': {0: 0.0}}
    ),
    # 7. Duplicate Predictions (Execution Accuracy = 0.0, as duplicates change the structure)
    (
        [[(1, 'Alice'), (2, 'Bob')]],
        [[(1, 'Alice'), (1, 'Alice'), (2, 'Bob')]],
        {'total_execution_accuracy': 0.0, 'individual_execution_accuracy': {0: 0.0}}
    ),
    # 8. Perfect Match, Multiple Queries (Execution Accuracy = 1.0)
    (
        [[(1, 'Alice'), (2, 'Bob')], [(1, 'Bob'), (2, 'Alice')]],
        [[(1, 'Alice'), (2, 'Bob')], [(1, 'Bob'), (2, 'Alice')]],
        {'total_execution_accuracy': 1.0, 'individual_execution_accuracy': {0: 1.0, 1: 1.0}}
    ),
    # 9. Match and No Match, Multiple Queries (Execution Accuracy = 0.5)
    (
        [[(1, 'Alice'), (2, 'Bob')], [(1, 'Bob'), (2, 'Alice')]],
        [[(1, 'Alice'), (2, 'Bob')], [(1, 'Bob'), (2, 'Alice'), (3, 'Alice')]],
        {'total_execution_accuracy': 0.5, 'individual_execution_accuracy': {0: 1.0, 1: 0.0}}
    ),
    # 10. No Match, Multiple Queries (Execution Accuracy = 0.0)
    (
        [[(1, 'Alice'), (2, 'Bob')], [(1, 'Bob'), (2, 'Alice')]],
        [[], [(1, 'Bob'), (2, 'Alice'), (3, 'Alice')]],
        {'total_execution_accuracy': 0.0, 'individual_execution_accuracy': {0: 0.0, 1: 0.0}}
    ),
], ids=[
    'full match', 'generated one diff', 'generated full diff', 'empty generated', 'empty gold',
    'some missing some extra', 'duplicate predictions', 'multiple', 'match and no match', 
    'multiple no match'
])
def test_execution_accuracy_score(golden, generated, expected_result):
    # Arrange + Act
    res = execution_accuracy(golden, generated)

    # Assert
    assert res == expected_result
