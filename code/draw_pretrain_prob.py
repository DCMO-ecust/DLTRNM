import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir('../data')
data = []
with open('train_loss.txt', 'r') as f:
    for line in f:
        row = line.strip().split()
        data.append(float(row[0]))
sorted_data = np.sort(data)
cumulative_prob = np.linspace(0, 1, len(sorted_data))
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
loss_threshold = 10**2
idx_loss = np.searchsorted(sorted_data, loss_threshold)
y_value_at_loss_threshold = cumulative_prob[idx_loss]
plt.step(sorted_data, cumulative_prob)
plt.plot([loss_threshold, loss_threshold], [0, y_value_at_loss_threshold], color='r', linestyle='--')

plt.plot([0, loss_threshold], [y_value_at_loss_threshold, y_value_at_loss_threshold], color='g', linestyle='--')


plt.text(0.01, y_value_at_loss_threshold-0.01, f'{y_value_at_loss_threshold:.3f}',
         color='g', verticalalignment='top', horizontalalignment='right')
plt.xscale('log')
plt.title('Cumulative distribution of Pre-training Loss', fontsize=16)
plt.xlabel('Loss', fontsize=16)
plt.ylabel('Cumulative probability', fontsize=16)
plt.ylim(0, 1)
plt.grid(True)
ax = plt.gca()
ax.tick_params(direction='in')
os.chdir('../figure')
plt.savefig('Train cumulative probability.svg', bbox_inches='tight')
# plt.savefig('Train Cumulative Probability.svg', bbox_inches='tight')
