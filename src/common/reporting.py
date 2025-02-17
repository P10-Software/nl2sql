import html
from src.core.base_model import NL2SQLModel
from src.common.logger import get_logger

logger = get_logger(__name__)


def create_report(models: list[NL2SQLModel]):
    # Ensure models contain analysis
    for model in models:
        if model.analysis is None:
            logger.error(
                f'Model: {model.__class__.__name__} did not contain any analysis, please run analysis first.')
            raise ValueError('Missing analysis results in model.')

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

    # Add each model's analysis results as a row in the table
    for model in models:
        model_name = model.__class__.__name__
        total_errors = model.analysis.get(
            'SQL mismatches', {}).get('total_errors', {})

        html_content += f"""
            <tr>
                <td>{model_name}</td>
                <td onclick="toggleDetails('{model_name}-execution_accuracy')">{model.analysis['execution accuracy']['total_execution_accuracy']:.2f}</td>
                <td onclick="toggleDetails('{model_name}-precision')">{model.analysis['precision']['total_precision']:.2f}</td>
                <td onclick="toggleDetails('{model_name}-recall')">{model.analysis['recall']['total_recall']:.2f}</td>
                <td onclick="toggleDetails('{model_name}-f1_score')">{model.analysis['f1 score']['total_f1']:.2f}</td>
                <td>{model.analysis.get('total sql queries', 'N/A')}</td>
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
            for idx, val in model.analysis['execution accuracy'].get('individual_execution_accuracy', {}).items()
        ]
        html_content += generate_details_row(f"{model_name}-execution_accuracy", "Execution Accuracy Details", [
                                             "Query Index", "Correct Execution"], exec_acc_data)

        # Generate precision breakdown
        precision_data = [
            (idx, f"{val:.2f}")
            for idx, val in model.analysis['precision'].get('individual_precisions', {}).items()
        ]
        html_content += generate_details_row(f"{model_name}-precision", "Precision Details", [
                                             "Query Index", "Precision"], precision_data)

        # Generate recall breakdown
        recall_data = [
            (idx, f"{val:.2f}")
            for idx, val in model.analysis['recall'].get('individual_recalls', {}).items()
        ]
        html_content += generate_details_row(f"{model_name}-recall", "Recall Details", [
                                             "Query Index", "Recall"], recall_data)

        # Generate F1 score breakdown
        f1_data = [
            (idx, f"{val:.2f}")
            for idx, val in model.analysis['f1 score'].get('individual_f1s', {}).items()
        ]
        html_content += generate_details_row(f"{model_name}-f1_score", "F1 Score Details", [
                                             "Query Index", "F1 Score"], f1_data)

        # Generate SQL mismatches
        sql_mismatch_data = [
            (
                html.escape(entry.get('generated_sql', 'N/A')),
                html.escape(entry.get('gold_sql', 'N/A')),
                ', '.join(entry.get('errors', {}).get(
                    'tables', {}).get('gold', [])) or '✅',
                ', '.join(entry.get('errors', {}).get(
                    'tables', {}).get('generated', [])) or '✅',
                ', '.join(set(entry.get('errors', {}).get('columns', {}).get('gold', [])) -
                          set(entry.get('errors', {}).get('columns', {}).get('generated', []))) or '✅',
                ', '.join(set(entry.get('errors', {}).get('tables', {}).get('generated', [])) -
                          set(entry.get('errors', {}).get('tables', {}).get('gold', []))) or '✅',
                ' | '.join(
                    f"{clause}: {errors.get('gold', 'N/A')} -> {errors.get('generated', 'N/A')}"
                    for clause, errors in entry.get('errors', {}).get('clauses', {}).items()
                ) or '✅',
                'Mismatch' if entry.get('errors', {}).get('distinct', {}).get('gold', False) != entry.get(
                    'errors', {}).get('distinct', {}).get('generated', False) else '✅'
            )
            for entry in model.analysis['SQL mismatches'].get('individual_errors', [])
        ]

        html_content += generate_details_row(
            f"{model_name}-sql_mismatches", "SQL Mismatch Breakdown",
            ["Generated Query", "Gold Query", "Missing Tables", "Extra Tables",
                "Missing Columns", "Extra Columns", "Clause Errors", "Distinct Mismatch"],
            sql_mismatch_data
        )

    html_content += """
        </table>
    </body>
    </html>
    """

    with open("report.html", "w", encoding="utf-8") as file:
        file.write(html_content)
