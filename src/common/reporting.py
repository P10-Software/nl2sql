import os
import html
import statistics
from collections import Counter, defaultdict
from json import load
import sql_metadata
from src.common.logger import get_logger
from src.core.evaluation_metrics import execution_accuracy, precision, recall, f1_score
from src.database.database import execute_query

logger = get_logger(__name__)


class Reporter:
    def __init__(self):
        self.results = []
        self.analysis = []
        self.report = None


    def generate_report(self, result_directory: str):
        """
        Executes generated and goal queries and creates results to be used for creating the report.

        Args:
        - result_directory: The directory path to find the json result files.
        """
        for result_file_name in os.listdir(result_directory):
            path = f"{result_directory}/{result_file_name}"

            if result_file_name == "report.html":
                continue

            with open(path, "r") as file_pointer:
                results = load(file_pointer)

            logger.info(f"Running results of database for {path}.")
            for res in results.values():
                if res['golden_query']:
                    res['golden_result'] = execute_query(res['golden_query'])
                else:
                    res['golden_result'] = None

                if res['generated_query']:
                    res['generated_result'] = execute_query(res['generated_query'])
                else:
                    res['generated_result'] = None

            logger.info(f"Executed all queries on the database for {path}.")

            file_name = result_file_name.split('.')[0] #remove .json
            self.add_result(results, file_name.split('_')[0], file_name.split('_')[1])

        self.create_report(result_directory)


    def add_result(self, result, name, run):
        """
        Adds a result set to the reporter.

        args:
        - result: The result from NL2SQLModel.run()
        """
        self.results.append(result)
        self._analyse(result, name, run)

    def _analyse(self, result, name, run) -> None:
        """
        Generates an analysis of the results.
        Runs metrics of EX, recall, precision and F1. Also analysis SQL errors, categorising by table, column and clause.
        """
        if result == {}:
            logger.error("Analysis called on empty result.")
            return
        golden_results = [res['golden_result']
                          for res in result.values()]
        generated_results = [res['generated_result']
                             for res in result.values()]
        nl_questions = [res['question']for res in result.values()]

        golden_sql = [res['golden_query'] for res in result.values()]
        generated_sql = [res['generated_query']
                         for res in result.values()]

        sql_errors = self._analyse_sql(golden_sql, generated_sql, nl_questions)

        precision_score = precision(golden_results, generated_results)
        recall_score = recall(golden_results, generated_results)

        self.analysis.append((name, run, {
            'execution accuracy': execution_accuracy(golden_results, generated_results),
            'precision': precision_score,
            'recall': recall_score,
            'f1 score': f1_score(precision_score, recall_score),
            'SQL mismatches': sql_errors,
            'total sql queries': len(generated_sql)
        }))

    def _analyse_sql(self, golden_sql_list, generated_sql_list, nl_question_list):
        """
        Finds and aggregated the error clauses for all SQL, comparing two lists of generated and gold sql queries.
        """
        error_counts = Counter({
            "table_errors": 0,
            "column_errors": 0,
            "clause_errors": 0,
            "distinct_errors": 0,
            "not_query_errors": 0,
            "abstention_errors": 0
        })
        individual_errors = []

        for golden_sql, generated_sql, nl_question in zip(golden_sql_list, generated_sql_list, nl_question_list):
            mismatches = self._extract_sql_mismatches(golden_sql, generated_sql)

            # Count table mismatches
            if mismatches['tables']['golden'] or mismatches['tables']['generated']:
                error_counts["table_errors"] += 1

            # Count column mismatches
            if mismatches['columns']['golden'] or mismatches['columns']['generated']:
                error_counts["column_errors"] += 1

            # Count clause mismatches
            clause_mismatch_count = sum(1 for _ in mismatches["clauses"])
            error_counts["clause_errors"] += clause_mismatch_count

            if mismatches['distinct']['golden'] != mismatches['distinct']['generated']:
                error_counts['distinct_errors'] += 1

            if mismatches['not_query']:
                error_counts['not_query_errors'] += 1

            if mismatches['abstention']:
                error_counts['abstention_errors'] += 1

            individual_errors.append({
                'nl_question': nl_question,
                'generated_sql': generated_sql,
                'golden_sql': golden_sql,
                'errors': mismatches
            })

        return {
            'total_errors': dict(error_counts),
            'individual_errors': individual_errors
        }

    def _extract_sql_mismatches(self, golden_sql, generated_sql):
        if golden_sql is None or generated_sql is None:
            return {
                'tables': {'golden': [], 'generated': []},
                'columns': {'golden': [], 'generated': []},
                'clauses': {},
                'distinct': {'golden': False, 'generated': False},
                'not_query': False,
                'abstention': False if golden_sql is generated_sql else True
            }

        golden_parser = sql_metadata.Parser(golden_sql)
        generated_parser = sql_metadata.Parser(generated_sql)
        try:
            mismatches = {
                'tables': {'golden': [], 'generated': []},
                'columns': {'golden': [], 'generated': []},
                'clauses': {},
                'distinct': {'golden': self._has_dictinct(golden_parser), 'generated': self._has_dictinct(generated_parser)},
                'not_query': False,
                'abstention': False
            }

            golden_tables = set(golden_parser.tables) if golden_parser.tables else set()
            generated_tables = set(
                generated_parser.tables) if generated_parser.tables else set()
            if golden_tables != generated_tables:
                mismatches['tables']['golden'] = golden_parser.tables
                mismatches['tables']['generated'] = generated_parser.tables

            golden_columns = set(self._extract_columns(golden_parser))
            generated_columns = set(self._extract_columns(generated_parser))
            if golden_columns != generated_columns:
                mismatches['columns']['golden'] = list(golden_columns)
                mismatches['columns']['generated'] = list(generated_columns)

            golden_clauses = self._extract_clauses(golden_parser)
            generated_clauses = self._extract_clauses(generated_parser)

            clause_errors = {}

            for clause in golden_clauses:
                golden_values = golden_clauses[clause]
                generated_values = generated_clauses.get(clause, [])

                if golden_values != generated_values:
                    clause_errors[clause] = {
                        "golden": golden_values,
                        "generated": generated_values,
                    }

            mismatches['clauses'] = clause_errors

            return mismatches

        except Exception:
            return {
                'tables': {'golden': [], 'generated': []},
                'columns': {'golden': [], 'generated': []},
                'clauses': {},
                'distinct': {'golden': False, 'generated': False},
                'not_query': True,
                'abstention': False
            }

    def _extract_clauses(self, parser):
        clauses = {'WHERE': [], 'JOIN': [], 'GROUPBY': [], 'ORDERBY': []}

        for token in parser.tokens:
            if token.is_keyword and token.normalized in ['WHERE', 'JOIN', 'GROUPBY', 'ORDERBY']:
                curr_tok = token.next_token
                clause_filter = []
                while curr_tok and not curr_tok.is_keyword:
                    if curr_tok.is_name and '.' in curr_tok.normalized:
                        clause_filter.append(curr_tok.normalized.split('.')[1])
                    else:
                        clause_filter.append(curr_tok.normalized)
                    curr_tok = curr_tok.next_token

                clauses[token.normalized].append(" ".join(clause_filter))

        return clauses

    def _has_dictinct(self, parser):
        for token in parser.tokens:
            if token.is_keyword and token.normalized == 'DISTINCT':
                return True
        return False

    def _extract_columns(self, parser, with_tables: bool = False):
        columns = []

        def get_only_columns(parser):
            for column in parser.columns_dict['select']:
                if '.' in column:
                    columns.append(column.split('.')[1])
                else:
                    columns.append(column)
            return columns

        if with_tables:
            if len(parser.tables) == 1:
                table = parser.tables[0]
                columns = get_only_columns(parser)
                return [table + col for col in columns]
            else:
                for token in parser.tokens:
                    if token.is_keyword and token.normalized in ['SELECT', 'DISTINCT']:
                        next_token = token.next_token
                        column_names = []
                        while next_token is not None:
                            if next_token.value not in [',', '.']:
                                column_names.append(next_token.value)
                            next_token = next_token.next_token
                            if next_token is not None and next_token.normalized == 'FROM':
                                columns.extend(
                                    [(next_token.next_token.value + '.' + s if '.' not in s else s) for s in column_names])
                                break
                return columns
        else:
            return get_only_columns(parser)

    def create_report(self, file_location: str):
        """
        Builds HTML report based on analysis from all models in param. All models need to have executed their analysis before feeding to method.

        args:
        - models (list): A list containing all models to build a report for.
        - file_location: Location where the report will be placed, only include filelocation, not file name ex: ./temp, not ./temp/report.html
        """
        if self.analysis is None:
            logger.error('At least one analysis needed to perform reporting')
            raise ValueError('Missing analysis result')

        html_content = """
        <html>
        <head>
            <title>NL2SQL Model Comparison</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                table { width: 100%; border-collapse: collapse; margin-top: 20px; }
                th, td, caption { border: 1px solid #ddd; padding: 8px; text-align: center; cursor: pointer; }
                th, caption { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .hidden { display: none; }
                .details-table { width: 80%; margin: 10px auto; border: 1px solid #ddd; }
            </style>
            <script>
                function toggleDetails(rowId) {
                    var detailsRow = document.getElementById(rowId);
                    detailsRow.style.display = (detailsRow.style.display === "none" || detailsRow.style.display === "") ? "table-row" : "none";
                }
            </script>
        </head>
        <body>
            <h1>NL2SQL Model Benchmark Report</h1>
        """

        experiments_dict = defaultdict(list)
        for name, run, results in self.analysis:
            experiments_dict[name].append((name, run, results))

        # Convert to list of lists
        experiments = list(experiments_dict.values())

        for experiment in experiments:
            model_name = experiment[0][0]
            experiment.sort(key=lambda x: x[1])

            agg_ex = []
            agg_recall = []
            agg_precision = []
            agg_f1 = []

            html_content += f"""
                <table>
                <caption><b>{model_name}</b></caption>
                <tr>
                    <th>Run</th>
                    <th>Execution Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1 Score</th>
                    <th>Total SQL Queries</th>
                    <th>SQL Mismatches</th>
                </tr>
            """
            for _, run, a in experiment:
                agg_ex.append((a['execution accuracy']['total_execution_accuracy']))
                agg_precision.append((a['precision']['total_precision']))
                agg_recall.append((a['recall']['total_recall']))
                agg_f1.append((a['f1 score']['total_f1']))

                total_errors = a.get(
                    'SQL mismatches', {}).get('total_errors', {})

                html_content += f"""
                    <tr>
                        <td>{run}</td>
                        <td onclick="toggleDetails('{model_name}-execution_accuracy')">{a['execution accuracy']['total_execution_accuracy']:.2f}</td>
                        <td onclick="toggleDetails('{model_name}-precision')">{a['precision']['total_precision']:.2f}</td>
                        <td onclick="toggleDetails('{model_name}-recall')">{a['recall']['total_recall']:.2f}</td>
                        <td onclick="toggleDetails('{model_name}-f1_score')">{a['f1 score']['total_f1']:.2f}</td>
                        <td>{a.get('total sql queries', 'N/A')}</td>
                        <td onclick="toggleDetails('{model_name}-sql_mismatches')">⚠️ {sum(total_errors.values()) if total_errors else 0} Errors</td>
                    </tr>
                """

                # Generate detailed breakdown tables
                def generate_details_row(row_id, title, headers, data):
                    return f"""
                    <tr id="{row_id}" class="hidden">
                        <td colspan="7">
                            <h3>{title}</h3>
                            <table class="details-table">
                                <tr>{''.join(f'<th>{h}</th>' for h in headers)}</tr>
                                {''.join(f"<tr>{''.join(f'<td>{html.escape(str(cell))}</td>' for cell in row)}</tr>" for row in data)}
                            </table>
                        </td>
                    </tr>
                    """ if data else ""

                # Generate execution accuracy breakdown
                exec_acc_data = [
                    (idx, '✅' if val else '❌')
                    for idx, val in a['execution accuracy'].get('individual_execution_accuracy', {}).items()
                ]
                html_content += generate_details_row(f"{model_name}-execution_accuracy", "Execution Accuracy Details", [
                                                    "Query Index", "Correct Execution"], exec_acc_data)

                # Generate precision breakdown
                precision_data = [
                    (idx, f"{val:.2f}")
                    for idx, val in a['precision'].get('individual_precisions', {}).items()
                ]
                html_content += generate_details_row(f"{model_name}-precision", "Precision Details", [
                                                    "Query Index", "Precision"], precision_data)

                # Generate recall breakdown
                recall_data = [
                    (idx, f"{val:.2f}")
                    for idx, val in a['recall'].get('individual_recalls', {}).items()
                ]
                html_content += generate_details_row(f"{model_name}-recall", "Recall Details", [
                                                    "Query Index", "Recall"], recall_data)

                # Generate F1 score breakdown
                f1_data = [
                    (idx, f"{val:.2f}")
                    for idx, val in a['f1 score'].get('individual_f1s', {}).items()
                ]
                html_content += generate_details_row(f"{model_name}-f1_score", "F1 Score Details", [
                                                    "Query Index", "F1 Score"], f1_data)

                # Generate SQL mismatches
                sql_mismatch_data = [
                    (
                        entry.get('nl_question', 'N/A'),
                        entry.get('golden_sql', 'N/A'),
                        entry.get('generated_sql', 'N/A'),
                        ', '.join(entry.get('errors', {}).get(
                            'tables', {}).get('golden', [])) or '✅',
                        ', '.join(entry.get('errors', {}).get(
                            'tables', {}).get('generated', [])) or '✅',
                        ', '.join(set(entry.get('errors', {}).get('columns', {}).get('golden', [
                        ])) - set(entry.get('errors', {}).get('columns', {}).get('generated', []))) or '✅',
                        ', '.join(set(entry.get('errors', {}).get('columns', {}).get('generated', [])) - set(entry.get('errors', {}).get('columns', {}).get('golden', [
                        ]))) or '✅',
                        ' | '.join(
                            f"{clause}:{errors.get('generated', 'N/A')}"
                            for clause, errors in entry.get('errors', {}).get('clauses', {}).items()
                        ) or '✅',
                        '❌' if entry.get('errors', {}).get('distinct', {}).get('golden', False) != entry.get(
                            'errors', {}).get('distinct', {}).get('generated', False) else '✅',
                        '❌' if entry.get('errors', {}).get('not_query') else '✅',
                        '❌' if entry.get('errors', {}).get('abstention') else '✅'
                    )
                    for entry in a['SQL mismatches'].get('individual_errors', [])
                ]

                html_content += generate_details_row(
                    f"{model_name}-sql_mismatches", "SQL Mismatch Breakdown",
                    ["NL Question", "Golden Query", "Generated Query", "Missing Tables", "Extra Tables",
                        "Missing Columns", "Extra Columns", "Clause Errors", "Distinct Mismatch", "Execution Failed", "Abstention Mismatch"],
                    sql_mismatch_data
                )

            if len(experiment) > 1:
                html_content += f"""
                    <tr>
                        <th>Average</th>
                        <th>{round(statistics.mean(agg_ex), 2)}</th>
                        <th>{round(statistics.mean(agg_precision), 2)}</th>
                        <th>{round(statistics.mean(agg_recall), 2)}</th>
                        <th>{round(statistics.mean(agg_f1), 2)}</th>
                        <th>Not Applicable</th>
                        <th>Not Applicable</th>
                    <tr>
                    <tr>
                        <th>Std Dev</th>
                        <th>{round(statistics.stdev(agg_ex), 2)}</th>
                        <th>{round(statistics.stdev(agg_precision), 2)}</th>
                        <th>{round(statistics.stdev(agg_recall), 2)}</th>
                        <th>{round(statistics.stdev(agg_f1), 2)}</th>
                        <th>Not Applicable</th>
                        <th>Not Applicable</th>
                    </tr>
                """

            html_content += "</table>"

        html_content += """
        </body>
        </html>
        """

        if not os.path.isdir(file_location):
            os.makedirs(file_location)
        with open(file_location + "/report.html", "w", encoding="utf-8") as file:
            file.write(html_content)
