from src.core.prompt_strategies import XiYanSQLPromptStrategy, Llama3PromptStrategy, DeepSeekPromptStrategy
import pytest

XIYAN_SQL_EXPECTED = """
    You are now a Postgres data analyst, and you are given a database schema as follows:
    【Schema】
    CREATE TABLE People (
        PersonID INT PRIMARY KEY AUTO_INCREMENT,
        FirstName VARCHAR(50) NOT NULL,
        LastName VARCHAR(50) NOT NULL,
        Age INT CHECK (Age >= 0),
        DateOfBirth DATE,
    );
    【Question】
    What is the age of bob?
    【Evidence】

    Please read and understand the database schema carefully, and generate an executable SQL based on the user's question and evidence. The generated SQL is protected by ```sql and ```.
"""

LLAMA_EXPECTED = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a helpful AI assistant translating natural language to Postgres queries.
    You have access to the following database:
    CREATE TABLE People (
        PersonID INT PRIMARY KEY AUTO_INCREMENT,
        FirstName VARCHAR(50) NOT NULL,
        LastName VARCHAR(50) NOT NULL,
        Age INT CHECK (Age >= 0),
        DateOfBirth DATE,
    );
    <|eot_id|><|start_header_id|>user<|end_header_id|>

    What is the age of bob?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

DEEPSEEK_EXPECTED = """
    You are tasked with translating a question to SQL that can be executed on a Postgres database.

    You have access to the following database: 
    CREATE TABLE People (
        PersonID INT PRIMARY KEY AUTO_INCREMENT,
        FirstName VARCHAR(50) NOT NULL,
        LastName VARCHAR(50) NOT NULL,
        Age INT CHECK (Age >= 0),
        DateOfBirth DATE,
    );

    The question is:
    What is the age of bob?
"""

@pytest.mark.parametrize(("prompt_class", "expected_prompt"), [
    (XiYanSQLPromptStrategy, XIYAN_SQL_EXPECTED),
    (Llama3PromptStrategy, LLAMA_EXPECTED),
    (DeepSeekPromptStrategy, DEEPSEEK_EXPECTED)
])
def test_prompt_strategies(prompt_class, expected_prompt):
    # arrange
    schema = """
        CREATE TABLE People (
            PersonID INT PRIMARY KEY AUTO_INCREMENT,
            FirstName VARCHAR(50) NOT NULL,
            LastName VARCHAR(50) NOT NULL,
            Age INT CHECK (Age >= 0),
            DateOfBirth DATE,
        );
    """
    question = "What is the age of bob?"
    sql_dialect = "Postgres"

    # act
    prompt_strategy = prompt_class(sql_dialect)
    prompt = prompt_strategy.get_prompt(schema, question)

    # assert
    assert "".join(prompt.split()) == "".join(expected_prompt.split())