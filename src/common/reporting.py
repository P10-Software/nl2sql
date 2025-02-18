import html
from src.common.logger import get_logger
from src.core.evaluation_metrics import execution_accuracy, precision, recall, f1_score
from collections import Counter
import sql_metadata
import os

logger = get_logger(__name__)


class Reporter:
    def __init__(self):
        self.results = []
        self.analysis = []
        self.report = None

    def add_result(self, result, name):
        """
        Adds a result set to the reporter.

        args:
        - result: The result from NL2SQLModel.run()
        """
        self.results.append(result)
        self._analyse(result, name)

    def _analyse(self, result, name) -> None:
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

        golden_sql = [res['golden_query'] for res in result.values()]
        generated_sql = [res['generated_query']
                         for res in result.values()]

        sql_errors = self._analyse_sql(golden_sql, generated_sql)

        self.analysis.append((name, {
            'execution accuracy': execution_accuracy(golden_results, generated_results),
            'precision': precision(golden_results, generated_results),
            'recall': recall(golden_results, generated_results),
            'f1 score': f1_score(golden_results, generated_results),
            'SQL mismatches': sql_errors,
            'total sql queries': len(generated_sql)
        }))

    def _analyse_sql(self, gold_sql_list, generated_sql_list):
        """
        Finds and aggregated the error clauses for all SQL, comparing two lists of generated and gold sql queries.
        """
        error_counts = Counter({
            "table_errors": 0,
            "column_errors": 0,
            "clause_errors": 0,
            "distinct_errors": 0
        })
        individual_errors = []

        for gold_sql, generated_sql in zip(gold_sql_list, generated_sql_list):
            mismatches = self._extract_sql_mismatches(gold_sql, generated_sql)

            # Count table mismatches
            if mismatches['tables']['gold'] or mismatches['tables']['generated']:
                error_counts["table_errors"] += 1

            # Count column mismatches
            if mismatches['columns']['gold'] or mismatches['columns']['generated']:
                error_counts["column_errors"] += 1

            # Count clause mismatches
            clause_mismatch_count = sum(1 for _ in mismatches["clauses"])
            error_counts["clause_errors"] += clause_mismatch_count

            if mismatches['distinct']['gold'] != mismatches['distinct']['generated']:
                error_counts['distinct_errors'] += 1

            individual_errors.append({
                'generated_sql': generated_sql,
                'gold_sql': gold_sql,
                'errors': mismatches
            })

        return {
            'total_errors': dict(error_counts),
            'individual_errors': individual_errors
        }

    def _extract_sql_mismatches(self, gold_sql, generated_sql):
        gold_parser = sql_metadata.Parser(gold_sql)
        generated_parser = sql_metadata.Parser(generated_sql)

        mismatches = {
            'tables': {'gold': [], 'generated': []},
            'columns': {'gold': [], 'generated': []},
            'clauses': {},
            'distinct': {'gold': self._has_dictinct(gold_parser), 'generated': self._has_dictinct(generated_parser)}
        }

        gold_tables = set(gold_parser.tables) if gold_parser.tables else set()
        generated_tables = set(
            generated_parser.tables) if generated_parser.tables else set()
        if gold_tables != generated_tables:
            mismatches['tables']['gold'] = gold_parser.tables
            mismatches['tables']['generated'] = generated_parser.tables

        gold_columns = set(self._extract_columns(gold_parser))
        generated_columns = set(self._extract_columns(generated_parser))
        if gold_columns != generated_columns:
            mismatches['columns']['gold'] = list(gold_columns)
            mismatches['columns']['generated'] = list(generated_columns)

        gold_clauses = self._extract_clauses(gold_parser)
        generated_clauses = self._extract_clauses(generated_parser)

        clause_errors = {}

        for clause in gold_clauses:
            gold_values = gold_clauses[clause]
            generated_values = generated_clauses.get(clause, [])

            if gold_values != generated_values:
                clause_errors[clause] = {
                    "gold": gold_values,
                    "generated": generated_values,
                }

        mismatches['clauses'] = clause_errors

        return mismatches

    def _extract_clauses(self, parser):
        clauses = {'WHERE': [], 'JOIN': [], 'GROUPBY': [], 'ORDERBY': []}

        for token in parser.tokens:
            if token.is_keyword and token.normalized in ['WHERE', 'JOIN', 'GROUPBY', 'ORDERBY']:
                curr_tok = token.next_token
                clause_filter = []
                while curr_tok and not curr_tok.is_keyword:
                    clause_filter.append(curr_tok.normalized)
                    curr_tok = curr_tok.next_token

                clauses[token.normalized].append(" ".join(clause_filter))

        return clauses

    def _has_dictinct(self, parser):
        for token in parser.tokens:
            if token.is_keyword and token.normalized == 'DISTINCT':
                return True
        return False

    def _extract_columns(self, parser):
        columns = []
        for token in parser.tokens:
            if token.is_keyword and token.normalized in ['SELECT', 'DISTINCT']:
                next = token.next_token
                while next and not next.is_keyword:
                    if next.normalized != ',':
                        columns.append(next.normalized.lower())
                    next = next.next_token
        return columns

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
                th, td { border: 1px solid #ddd; padding: 8px; text-align: center; cursor: pointer; }
                th { background-color: #f2f2f2; }
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
            <table>
                <tr>
                    <th>Model</th>
                    <th>Execution Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1 Score</th>
                    <th>Total SQL Queries</th>
                    <th>SQL Mismatches</th>
                </tr>
        """

        for name, a in self.analysis:
            model_name = name
            total_errors = a.get(
                'SQL mismatches', {}).get('total_errors', {})

            html_content += f"""
                <tr>
                    <td>{model_name}</td>
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
                    entry.get('gold_sql', 'N/A'),
                    entry.get('generated_sql', 'N/A'),
                    ', '.join(entry.get('errors', {}).get(
                        'tables', {}).get('gold', [])) or '✅',
                    ', '.join(entry.get('errors', {}).get(
                        'tables', {}).get('generated', [])) or '✅',
                    ', '.join(set(entry.get('errors', {}).get('columns', {}).get('gold', [
                    ])) - set(entry.get('errors', {}).get('columns', {}).get('generated', []))) or '✅',
                    ', '.join(set(entry.get('errors', {}).get('columns', {}).get('generated', [])) - set(entry.get('errors', {}).get('columns', {}).get('gold', [
                    ]))) or '✅',
                    ' | '.join(
                        f"{clause}:{errors.get('generated', 'N/A')}"
                        for clause, errors in entry.get('errors', {}).get('clauses', {}).items()
                    ) or '✅',
                    'Mismatch' if entry.get('errors', {}).get('distinct', {}).get('gold', False) != entry.get(
                        'errors', {}).get('distinct', {}).get('generated', False) else '✅'
                )
                for entry in a['SQL mismatches'].get('individual_errors', [])
            ]

            html_content += generate_details_row(
                f"{model_name}-sql_mismatches", "SQL Mismatch Breakdown",
                ["Gold Query", "Generated Query", "Missing Tables", "Extra Tables",
                    "Missing Columns", "Extra Columns", "Clause Errors", "Distinct Mismatch"],
                sql_mismatch_data
            )

        html_content += """
            </table>
        </body>
        </html>
        """

        if not os.path.isdir(file_location):
            os.makedirs(file_location)
        with open(file_location + "/report.html", "w", encoding="utf-8") as file:
            file.write(html_content)
