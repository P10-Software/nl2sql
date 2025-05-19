from src.core.extractive_schema_linking import load_schema_linker, predict_relevance_for_chunks
from src.core.schema_chunking import mschema_to_k_chunks, chunk_mschema
from tqdm import tqdm
import json
import statistics

NUMBER_OF_TRIALS = 10
EVALUATION_DATA_PATH = ".local/SchemaLinker/spider_exsl_all_to_single_test.json"
SCHEMA_LINKER_PATH = "models/EXSL/OmniSQL_7B_optimal_params_coarse_grained_schema_linker_spider.pth"
RESULT_DIRECTORY = ".local/experiments/schema_linking/exsl_omni/"
NUMBER_OF_CHUNKS = 1

def evaluate_extractive_schema_linking(schema_linker_path: str, dataset: list, chunk_amount: int = 0):
    schema_linker = load_schema_linker(schema_linker_path)
    sum_column_recall_at_5 = 0
    sum_column_recall_at_10 = 0
    sum_table_recall_at_5 = 0
    sum_table_recall_at_10 = 0
    sum_column_recall_100 = 0
    sum_table_recall_100 = 0
    report = []

    for example in tqdm(dataset):
        #chunks = mschema_to_k_chunks(example["schema"], schema_linker.tokenizer, chunk_amount)
        chunks = chunk_mschema(example["schema"], schema_linker, False)

        goal_columns = example["goal answer"]
        goal_tables = {column.split(" ")[0] for column in goal_columns}

        # Make relevance predictions
        predictions = predict_relevance_for_chunks(schema_linker, example["question"], chunks)
        columns, relevance = zip(*(predictions))
        columns, relevance = list(columns), list(relevance)
    
        # Evaluate column level recall@5
        relevant_columns_at_5 = [column for column in columns[:5] if column in goal_columns]
        column_recall_at_5 = len(relevant_columns_at_5) / len(goal_columns)
        sum_column_recall_at_5 += column_recall_at_5

        # Evaluate column level recall@10
        relevant_columns_at_10 = [column for column in columns[:10] if column in goal_columns]
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
        goal_columns_copy = set(goal_columns)
        goal_tables_copy = set(goal_tables)
        while goal_columns_copy and i < len(columns):
            if columns[i] in goal_columns_copy:
                goal_columns_copy.remove(columns[i])

            if columns[i].split(" ")[0] in goal_tables_copy:
                goal_tables_copy.remove(columns[i].split(" ")[0])
                if not goal_tables_copy:
                    table_recall_100 = i + 1

            i += 1

        column_recall_100 = i
        sum_table_recall_100 += table_recall_100
        sum_column_recall_100 += column_recall_100

        report.append({"question": example["question"], "goal columns": list(goal_columns), "top 5 columns": columns[:5], "top 5 relevance": relevance[:5], "column recall@5": column_recall_at_5, "table recall@5": table_recall_at_5,
                       "top 10 columns": columns[:10], "top 10 relevance": relevance[:10], "column recall@10": column_recall_at_10, "table recall@10": table_recall_at_10, "column recall 100 count": column_recall_100, "table recall 100 count": table_recall_100})

    report.append({"Dataset Size": len(dataset), "Total column recall@5": sum_column_recall_at_5 / len(dataset), "Total table recall@5": sum_table_recall_at_5 / len(dataset), "Total column recall@10": sum_column_recall_at_10 / len(dataset), 
                   "Total table recall@10": sum_table_recall_at_10 / len(dataset),  "Average column recall 100": sum_column_recall_100 / len(dataset), "Average table recall 100": sum_table_recall_100 / len(dataset)})
    return report

if __name__ == "__main__":
    column_recall_at_5_results = []
    column_recall_at_10_results = []
    table_recall_at_5_results = []
    table_recall_at_10_results = []
    column_recall_100_results = []
    table_recall_100_results = []

    for trial_num in tqdm(range(NUMBER_OF_TRIALS)):
        with open(EVALUATION_DATA_PATH, "r") as eval_file:
            eval_set = json.load(eval_file)
        report = evaluate_extractive_schema_linking(SCHEMA_LINKER_PATH, eval_set, NUMBER_OF_CHUNKS)

        column_recall_at_5_results.append(report[-1]["Total column recall@5"])
        column_recall_at_10_results.append(report[-1]["Total column recall@10"])
        table_recall_at_5_results.append(report[-1]["Total table recall@5"])
        table_recall_at_10_results.append(report[-1]["Total table recall@10"])
        column_recall_100_results.append(report[-1]["Average column recall 100"])
        table_recall_100_results.append(report[-1]["Average table recall 100"])

        with open(f"{RESULT_DIRECTORY}chunks_{NUMBER_OF_CHUNKS}_trial_{trial_num}.json", "w") as file:
            json.dump(report, file, indent=4)

    overall_report = {
        "Column recall@5": {"mean": statistics.mean(column_recall_at_5_results), "standard deviation": statistics.stdev(column_recall_at_5_results)},
        "Column recall@10": {"mean": statistics.mean(column_recall_at_10_results), "standard deviation": statistics.stdev(column_recall_at_10_results)},
        "Table recall@5": {"mean": statistics.mean(table_recall_at_5_results), "standard deviation": statistics.stdev(table_recall_at_5_results)},
        "Table recall@10": {"mean": statistics.mean(table_recall_at_10_results), "standard deviation": statistics.stdev(table_recall_at_10_results)},
        "Column recall 100": {"mean": statistics.mean(column_recall_100_results), "standard deviation": statistics.stdev(column_recall_100_results)},
        "Table recall 100": {"mean": statistics.mean(table_recall_100_results), "standard deviation": statistics.stdev(table_recall_100_results)}
    }

    with open(f"{RESULT_DIRECTORY}chunks_{NUMBER_OF_CHUNKS}_overview.json", "w") as file:
        json.dump(overall_report, file, indent=4)