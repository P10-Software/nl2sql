from collections import Counter
import pandas as pd


def precision(golden: list[list[tuple]], generated: list[list[tuple]]):
    """
    Calculates precision per question in sets, considering database query results.

    Args:
    - golden (list[list[tuple]]): List of lists, where each sublist contains tuples from executing the nth golden query.
    - generated (list[list[tuple]]): List of lists, where each sublist contains tuples from executing the nth generated query.

    Returns:
    - results (dict): A dictionary with:
        - "total_precision": The average precision across all queries.
        - "individual_precisions": A dictionary mapping query index to its precision score.
    """

    precision_scores = {}  # Store precision per query

    for idx, (gen_res, gold_res) in enumerate(zip(generated, golden)):

        gen_counts = Counter(gen_res)
        gold_counts = Counter(gold_res)

        true_positive = sum(min(gen_counts[row], gold_counts[row]) for row in gen_counts)
        false_positive = sum(gen_counts[row] - min(gen_counts[row], gold_counts[row]) for row in gen_counts)

        precision_scores[idx] = round((true_positive / (true_positive + false_positive)), 2) if (true_positive + false_positive) > 0 else 0.00

    total_precision = round((sum(precision_scores.values())/len(precision_scores)), 2) if len(precision_scores) > 0 else 0.00

    return {
        'total_precision': total_precision,
        'individual_precisions': precision_scores
    }


def recall(golden: list[list[tuple]], generated: list[list[tuple]]):
    """
    Calculates recall per question in sets, considering database query results.

    Args:
    - golden (list[list[tuple]]): List of lists, where each sublist contains tuples from executing the nth golden query.
    - generated (list[list[tuple]]): List of lists, where each sublist contains tuples from executing the nth generated query.

    Returns:
    - results (dict): A dictionary with:
        - "total_recall": The average recall across all queries.
        - "individual_recalls": A dictionary mapping query index to its recall score.
    """
    per_query_recall = {}

    for idx, (gen_results, gold_results) in enumerate(zip(generated, golden)):
        gold_counts = Counter(gold_results)
        gen_counts = Counter(gen_results)

        true_positive = sum(min(gen_counts[row], gold_counts[row]) for row in gold_counts)
        total_positive = sum(gold_counts.values())

        per_query_recall[idx] = round((true_positive / total_positive), 2) if total_positive > 0 else 0.00

    total_recall = round(sum(per_query_recall.values()) / len(per_query_recall), 2) if per_query_recall else 0.00

    return {
        'total_recall': total_recall,
        'individual_recalls': per_query_recall
    }


def f1_score(precision, recall):
    """
    Calculates F1-score per question in sets, considering database query results.

    Args:
    - golden (list[list[tuple]]): List of lists, where each sublist contains tuples from executing the nth golden query.
    - generated (list[list[tuple]]): List of lists, where each sublist contains tuples from executing the nth generated query.

    Returns:
    - results (dict): A dictionary with:
        - "total_f1": The average F1-score across all queries.
        - "individual_f1s": A dictionary mapping query index to its F1-score.
    """
    precision_scores = precision
    recall_scores = recall
    per_query_f1 = {}

    for idx in precision_scores["individual_precisions"]:
        p = precision_scores["individual_precisions"][idx]
        r = recall_scores["individual_recalls"][idx]

        f1 = round(((2 * p * r) / (p + r)), 2) if (p + r) > 0 else 0
        per_query_f1[idx] = f1

    total_f1 = round(sum(per_query_f1.values()) / len(per_query_f1), 2) if per_query_f1 else 0

    return {
        "total_f1": total_f1,
        "individual_f1s": per_query_f1
    }


def execution_accuracy(golden: list[list[tuple]], generated: list[list[tuple]]):
    """
    Implements the exact execution match, comparing the result of executing the goal query and generated query.

    Args:
        - golden (list[list[tuple]]): List of lists, where each sublist contains tuples from executing the nth golden query.
        - generated (list[list[tuple]]): List of lists, where each sublist contains tuples from executing the nth generated query.

    Returns;
    - Results (dict): A dictionary with:
        - "total_execution_accuracy": The average exact match score.
        - "individual_execution_accuracy": A dictionary mapping query index to its execution accuracy score.
    """

    per_query_execution_accuracy = {}

    for idx, (gold_res, gen_res) in enumerate(zip(golden, generated)):
        per_query_execution_accuracy[idx] = gold_res == gen_res

    total_execution_accuracy = sum(per_query_execution_accuracy.values()) / \
        len(per_query_execution_accuracy) if per_query_execution_accuracy else 0

    return {
        "total_execution_accuracy": total_execution_accuracy,
        "individual_execution_accuracy": per_query_execution_accuracy
    }
