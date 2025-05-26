import matplotlib.pyplot as plt

# Schema Linking Threshold values
thresholds = [0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

# Performance metrics
recall_ExSL_table = [99.9, 99.1, 97.1, 94.7, 90.3, 83.7, 76.6, 69.5]
size_reduction_ExSL_table = [3.28, 11.2, 19.6, 29.2, 39.8, 48.7, 55.9, 61.2]
recall_ExSL_column = [98.8, 95.9, 91.0, 83.3, 75.2, 67.5, 58.9, 51.8]
size_reduction_ExSL_column = [25.5, 47.8, 61.0, 69.8, 76.5, 81.5, 85.4, 88.1]


plt.figure(figsize=(9, 6))

# Plot original precision and recall
plt.plot(thresholds, recall_ExSL_table, 'k--^', label='Recall Table')
plt.plot(thresholds, size_reduction_ExSL_table, 'go-', label='Schema size reduction Table')
plt.plot(thresholds, recall_ExSL_column, 'bs-', label='Recall Column')
plt.plot(thresholds, size_reduction_ExSL_column, 'r:*', label='Schema size reduction Column')

# Axis labels and title
plt.xlabel('Schema Linking Threshold')
plt.ylabel('Performance')
plt.title('Balance between schema size reduction and recall based on threshold')

# Add legend and grid
plt.legend()
plt.grid(True)

# Show plot
plt.tight_layout()
plt.show()
