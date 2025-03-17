import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir('../data')
times = []
values = []
with open('interpolation_example.txt', 'r') as f:
    for line in f:
        row = line.strip().split()
        if row[0][0] == 'A':
            times.append(float(row[3]))
            values.append(float(row[4]))
        else:
            times.append(float(row[2]))
            values.append(float(row[3]))
x1 = times[0:7]
y1 = values[0:7]
x2 = times[8:]
y2 = values[8:]
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.scatter(x2, y2, color='blue', label='Interpolated data')
plt.scatter(x1, y1, color='red', label='Raw data')
plt.legend()
plt.xlabel('Time Point (min)')
plt.ylabel('Expression Level')
ax = plt.gca()
ax.tick_params(direction='in')
os.chdir('../figure')
plt.savefig('interpolation_example.svg', bbox_inches='tight')
# plt.savefig('interpolation_example.svg', bbox_inches='tight')
