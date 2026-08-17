import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

LOG_FILE = 'all_predictions_log.csv'
input_cols = ['Gender','AGE','Urea','Cr','HbA1c','Chol','TG','HDL','LDL','VLDL','BMI']

if not os.path.exists(LOG_FILE):
    print("No predictions logged yet. Please use the GUIs first.")
    exit(0)

# Expect confidence as last column
log = pd.read_csv(LOG_FILE, header=None, names=['timestamp','GUI_type'] + input_cols + ['PREDICTED_CLASS','CONFIDENCE'])

if log.empty:
    print("Log file exists but is empty.")
    exit(0)

# Round for robust matching
for col in input_cols:
    log[col] = log[col].astype(float).round(2)
log['test_id'] = log[input_cols].astype(str).agg('-'.join, axis=1)

# Collect high and low confidences per test case
results = {}
for tid, group in log.groupby('test_id'):
    preds = {r['GUI_type']: float(r['CONFIDENCE']) for _, r in group.iterrows()}
    results[tid] = {
        'high': preds.get('high', np.nan),
        'low': preds.get('low', np.nan)
    }

labels = []
high_conf = []
low_conf = []
for i, (tid, d) in enumerate(results.items()):
    labels.append(f'Test {i+1}')
    high_conf.append(d['high'])
    low_conf.append(d['low'])

if not labels:
    print("No test cases found in log file.")
    exit(0)

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(max(8, len(labels)), 6))
rects1 = ax.bar(x - width/2, high_conf, width, label='High Accuracy', color='green')
rects2 = ax.bar(x + width/2, low_conf, width, label='Low Accuracy', color='red')

ax.set_ylabel('Predicted Confidence')
ax.set_xlabel('Testcases')
ax.set_title('High vs Low GUI Confidence per Test Case')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45)
ax.set_ylim(0, 1.1)
ax.legend()
plt.tight_layout()
plt.show()