from src.core.extractive_schema_linking import load_schema_linker, predict_relevance_coarse
from tqdm import tqdm

def evaluate_extractive_schema_linking(schema_linker_path: str, dataset: list, split_count: int = 0):
    schema_linker = load_schema_linker(schema_linker_path)
    sum_column_recall_at_5 = 0
    sum_column_recall_at_10 = 0
    sum_table_recall_at_5 = 0
    sum_table_recall_at_10 = 0
    sum_column_recall_100 = 0
    sum_table_recall_100 = 0
    report = []

    for example in tqdm(dataset):
        # TODO: Make split of data

        goal_columns = example["goal answer"]
        goal_tables = {column.split(" ")[0] for column in goal_columns}

        # Make relevance predictions
        predictions = predict_relevance_coarse(schema_linker, example["question"], example["schema"])
        columns, relevance = zip(*(sorted(predictions.items(), reverse=True, key= lambda pair: pair[1])))
        columns, relevance = list(columns), list(relevance)
    
        # Evaluate column level recall@5
        relevant_columns_at_5 = {column for column in columns if column in goal_columns}[:5]
        column_recall_at_5 = len(relevant_columns_at_5) / len(goal_columns)
        sum_column_recall_at_5 += column_recall_at_5

        # Evaluate column level recall@10
        relevant_columns_at_10 = {column for column in columns if column in goal_columns}[:10]
        column_recall_at_10 = len(relevant_columns_at_10) / len(goal_columns)
        sum_column_recall_at_10 += column_recall_at_10

        # Evaluate table level recall@5
        relevant_tables_at_5 = {column.split(" ")[0] for column in relevant_columns_at_5 if column.split(" ")[0] in goal_tables}
        table_recall_at_5 = len(relevant_tables_at_5) / len(goal_tables)
        sum_table_recall_at_5 += table_recall_at_5

        # Evaluate table level recall@10
        relevant_tables_at_10 = {column.split(" ")[0] for column in relevant_columns_at_10 if column.split(" ")[0] in goal_tables}
        table_recall_at_10 = len(relevant_tables_at_10) / len(goal_tables)
        sum_table_recall_at_10 += table_recall_at_10

        # Determine element count where all columns and tables are found
        i = 0
        while goal_columns and i < len(goal_columns):
            if columns[i] in goal_columns:
                goal_columns.remove(columns[i])

            if columns[i].split(" ")[0] in goal_tables:
                goal_tables.remove(columns[i].split(" ")[0])
                if not goal_tables:
                    table_recall_100 = i + 1

            i += 1

        column_recall_100 = i
        sum_table_recall_100 += table_recall_100
        sum_column_recall_100 += column_recall_100

        report.append({"question": example["question"], "goal columns": list(goal_columns), "top 5 columns": relevant_columns_at_5, "top 5 relevance": relevance[:5], "column recall@5": column_recall_at_5, "table recall@5": table_recall_at_5,
                       "top 10 columns": relevant_columns_at_10, "top 10 relevance": relevance[:10], "column recall@10": column_recall_at_10, "table recall@10": table_recall_at_10, "column recall 100 count": column_recall_100, "table recall 100 count": table_recall_100})

    report.append({"Amount of questions": len(dataset), "Total column recall@5": sum_column_recall_at_5 / len(dataset), "Total table recall@5": sum_table_recall_at_5 / len(dataset), "Total column recall@10": sum_column_recall_at_10 / len(dataset), 
                   "Total table recall@5": sum_table_recall_at_10 / len(dataset),  "Average column recall 100": sum_column_recall_100 / len(dataset), "Average table recall 100": sum_table_recall_100 / len(dataset)})
    return report


        


if __name__ == "__main__":
    pass