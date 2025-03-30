import os
import sqlite3
from json import load
from tqdm import tqdm
from src.common.reporting import Reporter
from src.common.logger import get_logger

logger = get_logger(__name__)

# Directory containing json result files for resport
RESULT_DIR = "results"

def generate_report() -> None:
    reporter = Reporter()

    for result_file in tqdm(os.listdir(RESULT_DIR)):
        db_path = decide_db(result_file)

        if db_path == "none":
            continue

        if result_file == "report.html":
            continue

        reporter.add_result(evaluate_experiment(result_file, db_path), result_file.split('.')[0])

    reporter.create_report(RESULT_DIR)


def evaluate_experiment(file: str, db_path: str):
    with open(f"{RESULT_DIR}/{file}", "r") as file_pointer:
        results = load(file_pointer)

    for res in results.values():
        if res['golden_query']:
            res['golden_result'] = execute_query(res['golden_query'], db_path)
        else:
            res['golden_result'] = None

        if res['generated_query']:
            res['generated_result'] = execute_query(res['generated_query'], db_path)
        else:
            res['generated_result'] = None

    return results


def decide_db(file_name: str):
    if "ehrsql" in file_name.lower():
        logger.info(f"{file_name} uses mimic_iv DB")
        return ".local/mimic_iv.sqlite"
    elif "abbreviated" in file_name.lower():
        logger.info(f"{file_name} uses metadata abbreviated DB")
        return ".local/trial_metadata.sqlite"
    elif "natural" in file_name.lower():
        logger.info(f"{file_name} uses metadata natural DB")
        return ".local/trial_metadata_natural.sqlite"
    else:
        logger.info("Name does not indicate database")
        return "none"


def execute_query(query: str, db_path: str):
    conn = sqlite3.connect(f"{db_path}")
    cur = conn.cursor()

    result = []

    if conn:
        try:
            cur.execute(query)
            result = cur.fetchall()
        except Exception as e:
            logger.error(f"Error executing query on database: {e}")
        finally:
            cur.close()
            conn.close()

    return result


if __name__ == "__main__":
    generate_report()
