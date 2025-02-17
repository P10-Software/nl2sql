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
    per_query_precision = {}

    for idx, (gold_res, gen_res) in enumerate(zip(golden, generated)):
        gold_set = set(gold_res)
        gen_set = set(gen_res)

        tp = len(gen_set & gold_set)
        fp = len(gen_set - gold_set)

        precision_score = tp / (tp + fp) if (tp + fp) > 0 else 0
        per_query_precision[idx] = precision_score

    total_precision = sum(per_query_precision.values()) / \
        len(per_query_precision) if per_query_precision else 0

    return {
        "total_precision": total_precision,
        "individual_precisions": per_query_precision
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

    for idx, (gold_res, gen_res) in enumerate(zip(golden, generated)):
        gold_set = set(gold_res)
        gen_set = set(gen_res)

        tp = len(gen_set & gold_set)
        fn = len(gold_set - gen_set)

        recall_score = tp / (tp + fn) if (tp + fn) > 0 else 0
        per_query_recall[idx] = recall_score

    total_recall = sum(per_query_recall.values()) / len(per_query_recall) if per_query_recall else 0

    return {
        "total_recall": total_recall,
        "individual_recalls": per_query_recall
    }


def f1_score(golden: list[list[tuple]], generated: list[list[tuple]]):
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
    precision_scores = precision(golden, generated)
    recall_scores = recall(golden, generated)
    per_query_f1 = {}

    for idx in precision_scores["individual_precisions"]:
        p = precision_scores["individual_precisions"][idx]
        r = recall_scores["individual_recalls"][idx]

        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0
        per_query_f1[idx] = f1

    total_f1 = sum(per_query_f1.values()) / len(per_query_f1) if per_query_f1 else 0

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
