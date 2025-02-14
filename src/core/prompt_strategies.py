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
            你是一名{SQL_DIALECT}专家，现在需要阅读并理解下面的【数据库schema】描述，以及可能用到的【参考信息】，并运用{SQL_DIALECT}知识生成sql语句回答【用户问题】。
            【用户问题】
            {NL_QUESTION}

            【数据库schema】
            {DDL_INSTRUCTIONS}

            【参考信息】
            {EVIDENCE}

            【用户问题】
            {NL_QUESTION}

            ```sql
        """

    def get_prompt(self, schema, question):
        return self.prompt_template.format(SQL_DIALECT=self.sql_dialect, DDL_INSTRUCTIONS=schema, NL_QUESTION=question, EVIDENCE="")
    
class Llam3PromptStrategy(PromptStrategy):
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