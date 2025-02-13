from src.database.database import execute_query


def execution_accuracy(goal_query: str, generated_query: str) -> bool:
    """
    Implements the exact execution match, comparing the result of executing the goal query and generated query.
    
    Args:
    - goal_query: the expected query as a string.
    - generated_query: the LLM generated query to compare against the goal.

    Returns:
    - Returns True if the results are identical else False.

    """

    res_goal_query = execute_query(goal_query)
    res_generated_query = execute_query(generated_query)

    return True if res_generated_query == res_goal_query else False
