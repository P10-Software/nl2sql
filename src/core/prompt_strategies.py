from src.core.base_model import PromptStrategy

class DeepSeekPromptStrategy(PromptStrategy):
    def __init__(self, sql_dialect):
        self.sql_dialect = sql_dialect
        self.prompt_template = """
            You are tasked with translating a question to SQL that can be executed on a {SQL_DIALECT} database.

            You have access to the following database: 
            {DDL_INSTRUCTIONS}

            The question is:
            {NL_QUESTION}
        """

    def get_prompt(self, schema, question):
        return self.prompt_template.format(SQL_DIALECT=self.sql_dialect, DDL_INSTRUCTIONS=schema, NL_QUESTION=question)
    
class XiYanSQLPromptStrategy(PromptStrategy):
    def __init__(self, sql_dialect):
        self.sql_dialect = sql_dialect
        self.prompt_template = """
            You are now a {SQL_DIALECT} data analyst, and you are given a database schema as follows:

            【Schema】
            {DDL_INSTRUCTIONS}

            【Question】
            {NL_QUESTION}

            【Evidence】
            {EVIDENCE}

            Please read and understand the database schema carefully, and generate an executable SQL based on the user's question and evidence. The generated SQL is protected by ```sql and ```.
        """

    def get_prompt(self, schema, question):
        return self.prompt_template.format(SQL_DIALECT=self.sql_dialect, DDL_INSTRUCTIONS=schema, NL_QUESTION=question, EVIDENCE="")
    
class Llama3PromptStrategy(PromptStrategy):
    def __init__(self, sql_dialect):
        self.sql_dialect = sql_dialect
        self.prompt_template = """
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a helpful AI assistant translating natural language to {SQL_DIALECT} queries.
            You have access to the following database:
            {DDL_INSTRUCTIONS}
            <|eot_id|><|start_header_id|>user<|end_header_id|>

            {NL_QUESTION}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """

    def get_prompt(self, schema, question):
        return self.prompt_template.format(SQL_DIALECT=self.sql_dialect, DDL_INSTRUCTIONS=schema, NL_QUESTION=question)
    
class SQLCoderAbstentionPromptStrategy(PromptStrategy):
    def __init__(self, sql_dialect):
        self.sql_dialect = sql_dialect
        self.pre_sql_prompt_template = """
            ### Task
            Generate a {SQL_DIALECT} SQL query to answer [QUESTION]{NL_QUESTION}[/QUESTION]

            ### Instructions 
            - If you cannot answer the question with the available database schema, return 'I do not know'

            ### Database Schema
            The query will run on a database with the following schema:
            {DDL_INSTRUCTIONS}

            ### Answer
            Given the database schema, here is the SQL query that [QUESTION]{NL_QUESTION}[/QUESTION]
            [SQL]
        """
        self.post_sql_prompt_template = """
            #### Based on the question and predicted {SQL_DIALECT} SQL, are you sure the SQL below is correct? If you consider the SQL is correct, answer me with 'correct'. 
            If not, answer me with 'incorrect'. Only output your response without explanation.

            ### Database Schema
            The query will run on a database with the following schema:
            {DDL_INSTRUCTIONS}

            Question: {NL_QUESTION}
            Predicted SQL: {SQL}
            Answer:        
        """

    def get_prompt(self, schema, question, generated_sql= None):
        if not generated_sql:
            return self.pre_sql_prompt_template.format(SQL_DIALECT=self.sql_dialect, DDL_INSTRUCTIONS=schema, NL_QUESTION=question)
        else:
            return self.post_sql_prompt_template.format(SQL_DIALECT=self.sql_dialect, DDL_INSTRUCTIONS=schema, NL_QUESTION=question, SQL=generated_sql)