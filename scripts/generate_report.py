import os
from json import load, dump
# from dotenv import load_dotenv
import dotenv
from src.database.database import execute_query

dotenv_file = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv_file)

ORIGINAL_DB_PATH = os.environ["DB_PATH"]

# MODEL = os.getenv('MODEL')
# DB_PATH = os.getenv('DB_PATH')

def generate_report() -> None:
    print(os.environ["DB_PATH"])

    os.environ['DB_PATH'] = "NEW_PATH_naME"
    dotenv.set_key(dotenv_file, "DB_PATH", os.environ['DB_PATH'])

    execute_query("test")

    os.environ['DB_PATH'] = "NEW_PATH_WITH_NATURAL"
    dotenv.set_key(dotenv_file, "DB_PATH", os.environ['DB_PATH'])

    execute_query("test")

    set_env_variable("DB_PATH", ORIGINAL_DB_PATH)

    execute_query("test")


def set_env_variable(key: str, value: str) -> None:
    os.environ[key] = value
    dotenv.set_key(dotenv_file, key, os.environ[key])
    print(f"Changed env variable {key} to {value}")


if __name__ == "__main__":
    generate_report()
