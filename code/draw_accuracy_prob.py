import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir('../data')
accuracy_data = []
with open('valid_fine_tuned_output.csv', 'r') as f:
    for line in f:
        row = line.strip().split(',')
        if row[0] == 'NO':
            continue
        accuracy_data.append(1-float(row[7])/100)
        # testing_loss_data.append(float(row[19]))
training_sorted_data = np.sort(accuracy_data)
training_cumulative_prob = np.linspace(0, 1, len(accuracy_data))
plt.step(training_sorted_data, training_cumulative_prob)
loss_threshold = 0.8
idx_loss = np.searchsorted(training_sorted_data, loss_threshold)
y_value_at_loss_threshold = training_cumulative_prob[idx_loss]
plt.plot([loss_threshold, loss_threshold], [0, y_value_at_loss_threshold], color='r', linestyle='--')
plt.plot([0, loss_threshold], [y_value_at_loss_threshold, y_value_at_loss_threshold], color='g', linestyle='--')
plt.text(0, y_value_at_loss_threshold-0.01, f'{y_value_at_loss_threshold:.3f}',
         color='g', verticalalignment='top', horizontalalignment='left')
loss_threshold3 = 0.9
idx_loss = np.searchsorted(training_sorted_data, loss_threshold3)
y_value_at_loss_threshold3 = training_cumulative_prob[idx_loss]
plt.plot([loss_threshold3, loss_threshold3], [0, y_value_at_loss_threshold3], color='r', linestyle='--')
plt.plot([0, loss_threshold3], [y_value_at_loss_threshold3, y_value_at_loss_threshold3], color='g', linestyle='--')
plt.text(0, y_value_at_loss_threshold3-0.01, f'{y_value_at_loss_threshold3:.3f}',
         color='g', verticalalignment='top', horizontalalignment='left')
plt.text(loss_threshold3, -0.01, f'{loss_threshold3}',
         color='r', verticalalignment='top', horizontalalignment='center')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.title('Cumulative distribution of Subnetwork Accuracy')
plt.xlabel('Accuracy')
plt.ylabel('Cumulative Probability')
plt.grid(True)
plt.ylim(0, 1)
plt.xlim(0, 1)
ax = plt.gca()
ax.tick_params(direction='in')
os.chdir('../figure')
plt.savefig('Subnetwork Accuracy Cumulative Probability.svg', bbox_inches='tight')
# plt.savefig('Accuracy Cumulative Probability.svg', bbox_inches='tight')
