import json
from collections import Counter
from statistics import mean, stdev

with open(".local/experiment3/metadata/tables_per_1_overview.json", "r") as file:
   results = json.load(file)

cdw_exec_num_counts = []
trial_metadata_version_counts = []
trial_definition_id_counts = []
trial_id_counts = []
cdw_partition_key_counts = []
error_mark_counts = []
# Count occurrences
for result in results[:-1]:
    columns = [entry.split()[1] for entry in result["predicted tables"]]
    column_counts = Counter(columns)
    cdw_exec_num_counts.append(column_counts.get("cdw_exec_num", 0))
    trial_metadata_version_counts.append(column_counts.get("trial_metadata_version", 0))
    trial_definition_id_counts.append(column_counts.get("trial_definition_id", 0))
    trial_id_counts.append(column_counts.get("trial_id", 0))
    cdw_partition_key_counts.append(column_counts.get("cdw_partition_key", 0))
    error_mark_counts.append(column_counts.get("error_mark", 0))

# Function to summarize stats
def print_stats(name, data, original_count):
    if len(data) >= 2:
        print(f"{name}: mean = {mean(data):.2f}, stdev = {stdev(data):.2f}, count in original schema: {original_count}")
    elif len(data) == 1:
        print(f"{name}: mean = {mean(data):.2f}, stdev = N/A (only one value), count in original schema: {original_count}")
    else:
        print(f"{name}: no data, count in original schema: {original_count}")

# Print stats for each column
print_stats("cdw_exec_num", cdw_exec_num_counts, 51)
print_stats("trial_metadata_version", trial_metadata_version_counts, 38)
print_stats("trial_definition_id", trial_definition_id_counts, 38)
print_stats("trial_id", trial_id_counts, 38)
print_stats("cdw_partition_key", cdw_partition_key_counts, 36)
print_stats("error_mark", error_mark_counts, 35)