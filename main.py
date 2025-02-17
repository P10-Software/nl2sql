from src.core.model_implementations import LlamaModel, DeepSeekLlamaModel, DeepSeekQwenModel, XiYanSQLModel
from src.core.prompt_strategies import Llama3PromptStrategy, DeepSeekPromptStrategy, XiYanSQLPromptStrategy
from src.database.setup_database import get_conn

SQL_DIALECT = "postgres"
SCHEMA_SIZE = "full"

if __name__ == "__main__":
    connection = get_conn()
    dataset = [] # Insert dataset of the format [{question: QUESTION, golden_query: SQL}]
    prompt_strategy = XiYanSQLPromptStrategy(SQL_DIALECT)
    model = XiYanSQLModel(connection, dataset, prompt_strategy)
    model.run(SCHEMA_SIZE)
    
    # TODO: Do analysis and reporting