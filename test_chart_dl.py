import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the benchmark results
df = pd.read_csv('results/benchmark_results.csv')

# Create pivot table with Model Group (matching the chart in the image)
pivot = df.pivot_table(index='Model Group', columns='Dataset', values='AUC')

print('Pivot index:', list(pivot.index))
print('\nChecking each label:')
for i, label in enumerate(pivot.index):
    print(f'{i}: "{label}" (len={len(label)}, type={type(label).__name__})')
    print(f'   repr: {repr(label)}')
    print(f'   str(): {str(label)}')

# Create the chart
fig, ax = plt.subplots(figsize=(10, 6))

for col in pivot.columns:
    lbl = str(col)
    if lbl.lower().endswith('.csv'):
        lbl = lbl[:-4]
    ax.plot(pivot.index, pivot[col], marker="o", label=lbl)

ax.set_title("Model comparison (AUC)")
ax.set_xlabel("Model")
ax.set_ylabel("AUC")
ax.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc="upper left")
ax.set_xticks(range(len(pivot.index)))

# This is the key line - check what happens with the labels
labels = [str(x) for x in pivot.index]
print('\nX-tick labels:')
for i, lbl in enumerate(labels):
    print(f'{i}: "{lbl}"')

ax.set_xticklabels(labels, rotation=45, ha="right")
fig.tight_layout()

# Save the chart
fig.savefig('test_chart_output.png', dpi=300, bbox_inches='tight')
print('\nChart saved to test_chart_output.png')
plt.close(fig)
