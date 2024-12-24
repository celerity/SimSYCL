"""
Renders the ratios from `cts_state.csv` as `resources/cts_state.svg`.
"""

import os

import pandas as pd
from matplotlib import pyplot as plt

os.chdir(os.path.join(os.path.dirname(__file__), os.path.pardir))

state = pd.read_csv('ci/cts_state.csv', delimiter=';')
counts = state.groupby('status').agg(count=('suite', 'size'))['count'].to_dict()

labels = ['passed', 'run failed', 'build failed', 'not applicable']
colors = ['#4a0', '#fa0', '#e44', '#aaa']

plt.rcParams['svg.fonttype'] = 'none'

fig, ax = plt.subplots(figsize=(8, 0.6))
left = 0
for l, c in zip(labels, colors):
    n = counts[l]
    ax.barh(0, n, left=left, color=c, label=l)
    ax.text(left + n/2, 0, str(n), ha='center', va='center', weight='bold')
    left += n
ax.set_xlim(0, left)
ax.axis('off')
ax.set_title('SimSYCL spec conformance by number of CTS categories')

fig.legend(loc='lower center', ncols=len(labels),
           bbox_to_anchor=(0, -0.4, 1, 0.5), frameon=False)
fig.savefig('resources/cts_state.svg', bbox_inches='tight')
