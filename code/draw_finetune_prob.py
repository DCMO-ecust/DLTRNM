import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir('../data')
training_loss_data = []
testing_loss_data = []

with open('processed_fine_tune_result.txt', 'r') as f:
    for line in f:
        row = line.strip().split()
        training_loss_data.append(float(row[0]))
        testing_loss_data.append(float(row[1]))

training_sorted_data = np.sort(training_loss_data)
training_cumulative_prob = np.linspace(0, 1, len(training_sorted_data))

testing_sorted_data = np.sort(testing_loss_data)
testing_cumulative_prob = np.linspace(0, 1, len(testing_sorted_data))

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

plt.step(training_sorted_data, training_cumulative_prob, label='Training', color='b')

plt.step(testing_sorted_data, testing_cumulative_prob, label='Testing', color='orange')

def find_cumulative_prob(data, threshold):
    sorted_data = np.sort(data)
    idx = np.searchsorted(sorted_data, threshold)
    return np.interp(threshold, sorted_data, np.linspace(0, 1, len(sorted_data)))

training_y_at_0_2 = find_cumulative_prob(training_loss_data, 0.2)
testing_y_at_0_2 = find_cumulative_prob(testing_loss_data, 0.2)

plt.plot([0, 0.2], [training_y_at_0_2, training_y_at_0_2], color='b', linestyle='--')
plt.plot([0, 0.2], [testing_y_at_0_2, testing_y_at_0_2], color='orange', linestyle='--')

plt.plot([0.2, 0.2], [0, training_y_at_0_2], color='r', linestyle='--')
plt.text(10e-12, training_y_at_0_2-0.01, f'{training_y_at_0_2:.3f}',
         color='b', verticalalignment='top', horizontalalignment='left')
plt.text(10e-12, testing_y_at_0_2-0.01, f'{testing_y_at_0_2:.3f}',
         color='orange', verticalalignment='top', horizontalalignment='left')
plt.text(0.2, -0.01, f'{0.2}',
         color='r', verticalalignment='top', horizontalalignment='center')

plt.legend()
plt.xscale('log')
plt.title('Cumulative distribution of Fine-tuning Loss', fontsize=16)
plt.xlabel('Loss', fontsize=16)
plt.ylabel('Cumulative probability', fontsize=16)

plt.ylim(0, 1)
ax = plt.gca()
ax.tick_params(direction='in')
plt.grid(True)

os.chdir('../figure')
plt.savefig('Fine tune Cumulative Probability.svg', bbox_inches='tight')
