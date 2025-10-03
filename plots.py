from sklearn.metrics import roc_curve, auc
import numpy as np
import csv
import plotly.graph_objects as go

# File path
file_path_300 = r'C:\Users\nandia.lelova\PycharmProjects\PythonProject\Scenario 1\AUC_Scenario_1_results_N50.csv'

def data_preds(path):
    first_column, second_column, third_column, forth_column = [], [], [], []
    with open(path, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            first_column.append(float(row[0]))
            second_column.append(float(row[1]))
            third_column.append(float(row[2]))
            forth_column.append(float(row[3]))
    return first_column, second_column, third_column, forth_column

# --- ROC with confidence intervals ---
first_column, second_column, third_column, forth_column = data_preds(file_path_300)
true_labels =  [0] * 100 + [0] * 100 + [0] * 100 + [1] * 100
pred_prob = first_column + second_column + third_column + forth_column

y_test = np.array(true_labels)
y_pred_prob = np.array(pred_prob)

fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Bootstrap for CI
n_bootstraps = 1000
rng = np.random.RandomState(42)

tprs_bootstrap = []
aucs_bootstrap = []
mean_fpr = np.linspace(0, 1, 100)

for i in range(n_bootstraps):
    indices = rng.randint(0, len(y_test), len(y_test))
    if len(np.unique(y_test[indices])) < 2:
        continue
    y_test_boot = y_test[indices]
    y_pred_prob_boot = y_pred_prob[indices]
    fpr_boot, tpr_boot, _ = roc_curve(y_test_boot, y_pred_prob_boot)
    tpr_interp = np.interp(mean_fpr, fpr_boot, tpr_boot)
    tprs_bootstrap.append(tpr_interp)
    aucs_bootstrap.append(auc(fpr_boot, tpr_boot))

tprs_bootstrap = np.array(tprs_bootstrap)
mean_tpr = tprs_bootstrap.mean(axis=0)
mean_auc = np.mean(aucs_bootstrap)

alpha = 0.95
lower_auc = np.percentile(aucs_bootstrap, (1 - alpha) / 2 * 100)
upper_auc = np.percentile(aucs_bootstrap, (1 + alpha) / 2 * 100)
lower_tpr = np.percentile(tprs_bootstrap, (1 - alpha) / 2 * 100, axis=0)
upper_tpr = np.percentile(tprs_bootstrap, (1 + alpha) / 2 * 100, axis=0)

# --- FPR/TPR for 3 methods (one point per method) ---
# Replace these with your actual single FPR/TPR values per method
tpr_methods = [0.91, 0.91, 0.91, 0.92]   # Method 1, 2, 3
fpr_methods = [0.825, 0.86, 0.87, 0.88]  # Method 1, 2, 3

method_names = ["Bias-test δ=0.01", "Bias-test δ=0.03", "Bias-test δ=0.05", "Bias-test δ=0.1"]

colors = ["purple" ,"green", "red", "yellow"]

# --- Plot ROC curve ---
fig = go.Figure()

# Main ROC curve
fig.add_trace(
    go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f"AUC({roc_auc:.3f})",
        line=dict(color='navy', width=2),
    )
)

# Confidence interval
fig.add_trace(
    go.Scatter(
        x=np.concatenate([mean_fpr, mean_fpr[::-1]]),
        y=np.concatenate([upper_tpr, lower_tpr[::-1]]),
        fill='toself',
        fillcolor='rgba(0, 176, 246, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% CI',
        showlegend=True
    )
)

# Random baseline
fig.add_trace(
    go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(color='red', dash='dash')
    )
)

# Method points
for i in range(4):
    fig.add_trace(
        go.Scatter(
            x=[fpr_methods[i]],
            y=[tpr_methods[i]],
            mode='markers',
            name=method_names[i],
            marker=dict(symbol='square', size=18, color=colors[i])
        )
    )

# Add Ne annotation inside the plot
fig.add_annotation(
    x=0.5,       # centered horizontally
    y=1.08,      # slightly above the plot (outside axes)
    xref="paper",
    yref="paper",
    text="Ne=50",
    showarrow=False,
    font=dict(size=38, family="Times New Roman", color="black"),
    align="center"
)

# Layout
fig.update_layout(
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    legend=dict(x=0.75, y=0.05),
    template='plotly_white',
    width=800,
    height=600,
    xaxis_title_font=dict(size=38, color='black', family='Times New Roman'),
    yaxis_title_font=dict(size=38, color='black', family='Times New Roman'),
    legend_font=dict(size=28, color='black', family='Times New Roman'),
    font=dict(size=28, color='black', family='Times New Roman')
)

fig.show()

