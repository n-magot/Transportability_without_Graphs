import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# -------------------------------
# File paths
# -------------------------------
file_paths = {
    50: r'C:\Users\nandia.lelova\PycharmProjects\PythonProject\experiments_m_bias_Ne50_alpha0.99.csv',
    100: r'C:\Users\nandia.lelova\PycharmProjects\PythonProject\experiments_m_bias_Ne100_alpha0.99.csv',
    300: r'C:\Users\nandia.lelova\PycharmProjects\PythonProject\experiments_m_bias_Ne300_alpha0.99.csv'
}

# -------------------------------
# Function to read CSV
# -------------------------------
def data_preds(path):
    with open(path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # skip header
        columns = list(zip(*[[float(val) for val in row] for row in reader]))
    return columns

# -------------------------------
# Load and prepare data
# -------------------------------
algorithms = ['FindsABS',
              r'$D_e$',
              r'$D_o^*$',
              r'$D_e + D_o^*$']   # instead of 'Both'

all_dfs = []
for Ne, path in file_paths.items():
    F1, F2, F3, F4 = data_preds(path)
    data_lists = [F1, F2, F3, F4]
    for alg, values in zip(algorithms, data_lists):
        df = pd.DataFrame({
            'Ne': [Ne] * len(values),
            'Algorithm': [alg] * len(values),
            'Binary cross-entropy': values
        })
        all_dfs.append(df)

data = pd.concat(all_dfs, ignore_index=True)

# Ensure a proper order for algorithms
alg_order = [r'$D_e + D_o^*$', r'$D_e$', r'$D_o^*$', 'FindsABS']
data['Algorithm'] = pd.Categorical(data['Algorithm'], categories=alg_order, ordered=True)
data = data.sort_values(['Ne', 'Algorithm'])


# -------------------------------
# Summary statistics
# -------------------------------
summary = data.groupby(['Ne', 'Algorithm']).agg(
    mean_mse=('Binary cross-entropy', 'mean'),
    std_mse=('Binary cross-entropy', 'std'),
    count=('Binary cross-entropy', 'count')
).reset_index()
summary['sem_mse'] = summary['std_mse'] / np.sqrt(summary['count'])
summary['ci95'] = 1.96 * summary['sem_mse']  # 95% confidence interval

# -------------------------------
# Plotting setup
# -------------------------------
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']

palette = sns.color_palette('tab10', n_colors=len(alg_order))
custom_colors = {alg: palette[i] for i, alg in enumerate(alg_order)}

marker_styles = {'FindsABS': 'o', r'$D_e$': 's',
                 r'$D_o^*$': 'D', r'$D_e + D_o^*$': '^'}

# Define line thickness for each algorithm
line_widths = {'FindsABS': 3, r'$D_e$': 3, r'$D_o^*$': 3, r'$D_e + D_o^*$': 3}

# -------------------------------
# Plot figure
# -------------------------------
fig, ax = plt.subplots(figsize=(10.95, 6.74), dpi=100)

for alg in alg_order:
    subset = summary[summary['Algorithm'] == alg]
    ax.errorbar(
        subset['Ne'], subset['mean_mse'], yerr=subset['ci95'],
        fmt=marker_styles[alg] + '-',  # keep marker style
        capsize=6, label=alg,
        color=custom_colors[alg],
        linewidth=line_widths[alg],
        markersize=10,
        elinewidth=2
    )

ax.set_xlabel('Ne', fontsize=36)
ax.set_ylabel('Binary cross-entropy', fontsize=36)
ax.tick_params(axis='both', labelsize=30)
ax.set_xticks([50, 100, 300])
ax.set_yticks(np.arange(0.3, 0.7, 0.05))

ax.legend(
    title='',
    fontsize=30,

    frameon=True,
    fancybox=True,
    shadow=True
)
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()


# -------------------------------
# Save figure
# -------------------------------
# output_path = r'C:\Users\nandia.lelova\PycharmProjects\PythonProject\binary_cross_entropy_plot.png'
# fig.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()
