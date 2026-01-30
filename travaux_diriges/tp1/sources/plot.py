import matplotlib.pyplot as plt
import pandas as pd

# Data input
data = {
    'Processors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    'Metric': [2450.21, 3865.6, 5154.61, 6391.12, 7850.89, 8658.66, 9607.88, 9576.1, 10810, 10290.1, 8106.77, 11253, 11021.9, 10435.8, 11263.4, 10276.8]
}

df = pd.DataFrame(data)

# Calculate Speedup
# Speedup = Value_N / Value_1
base_metric = df['Metric'].iloc[0]
df['Speedup'] = df['Metric'] / base_metric

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(df['Processors'], df['Speedup'], marker='o', label='Actual Speedup', linewidth=2)
# Add ideal line for reference (y=x)
plt.plot(df['Processors'], df['Processors'], linestyle='--', color='gray', label='Ideal Linear Speedup')

plt.title('Speedup Curve')
plt.xlabel('Number of Units (e.g. Processors)')
plt.ylabel('Speedup')
plt.grid(True)
plt.legend()
plt.xticks(df['Processors'])

plt.savefig('speedup_curve.png')