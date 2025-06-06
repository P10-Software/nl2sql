import json
from collections import Counter
from statistics import mean, stdev

with open(".local/experiment3/metadata/tables_per_1_overview.json", "r") as file:
    results = json.load(file)

# Target repeated columns
target_columns = [
    "cdw_exec_num",
    "trial_metadata_version",
    "trial_definition_id",
    "trial_id",
    "cdw_partition_key",
    "error_mark",
]

# Initialize counters for each target column
column_counts_dict = {col: [] for col in target_columns}
# List to store repeated column percentages
repeated_col_percentages = []

# Process each result
for result in results[:-1]:
    columns = [entry.split()[1] for entry in result["predicted tables"]]
    column_counts = Counter(columns)

    # Record counts for each target column
    for col in target_columns:
        column_counts_dict[col].append(column_counts.get(col, 0))

    # Calculate repeated columns % of total columns
    total_columns = len(columns)
    repeated_columns_count = sum(column_counts.get(col, 0) for col in target_columns)

    if total_columns > 0:
        repeated_percentage = (repeated_columns_count / total_columns) * 100
        repeated_col_percentages.append(repeated_percentage)
    else:
        repeated_col_percentages.append(0.0)  # handle edge case if no columns

# Function to summarize stats
def print_stats(name, data, original_count=None):
    if len(data) >= 2:
        print(f"{name}: mean = {mean(data):.2f}, stdev = {stdev(data):.2f}", end="")
    elif len(data) == 1:
        print(f"{name}: mean = {mean(data):.2f}, stdev = N/A (only one value)", end="")
    else:
        print(f"{name}: no data", end="")

    if original_count is not None:
        print(f", count in original schema: {original_count}")
    else:
        print()

# Print stats for each target column
original_counts = {
    "cdw_exec_num": 51,
    "trial_metadata_version": 38,
    "trial_definition_id": 38,
    "trial_id": 38,
    "cdw_partition_key": 36,
    "error_mark": 35,
}

for col in target_columns:
    print_stats(col, column_counts_dict[col], original_counts[col])

# Print stats for repeated columns percentage
print_stats("Repeated columns % of total columns", repeated_col_percentages)
