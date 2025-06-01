import matplotlib.pyplot as plt

# Schema Linking Threshold values
thresholds = [0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

# Performance metrics
recall_ExSL_table = [0.999, 0.991, 0.971, 0.947, 0.903, 0.837, 0.766, 0.695]
size_reduction_ExSL_table = [0.0328, 0.112, 0.196, 0.292, 0.398, 0.487, 0.559, 0.612]
recall_ExSL_column = [0.988, 0.959, 0.910, 0.833, 0.752, 0.675, 0.589, 0.518]
size_reduction_ExSL_column = [0.255, 0.478, 0.610, 0.698, 0.765, 0.815, 0.854, 0.881]
precision_ExSL_table = [0.415, 0.458, 0.511, 0.584, 0.667, 0.715, 0.734, 0.705]
precision_ExSL_column = [0.195, 0.270, 0.350, 0.419, 0.491, 0.545, 0.559, 0.567]

plt.figure(figsize=(9, 6))

# Plot original precision and recall
plt.plot(thresholds, recall_ExSL_table, 'k--^', label='Recall Table')
plt.plot(thresholds, recall_ExSL_column, 'b--s', label='Recall Column')

plt.plot(thresholds, precision_ExSL_table, 'm:d', label='Precision Table')
plt.plot(thresholds, precision_ExSL_column, 'c:h', label='Precision Column')

plt.plot(thresholds, size_reduction_ExSL_table, 'g-o', label='Schema size reduction Table')
plt.plot(thresholds, size_reduction_ExSL_column, 'r-*', label='Schema size reduction Column')

# Axis labels and title
plt.xlabel('Schema Linking Threshold', fontsize=16)
plt.ylabel('Performance', fontsize=16)

# Add legend and grid
plt.legend(fontsize=16)
plt.grid(True)

# Show plot
plt.tight_layout()
plt.show()
