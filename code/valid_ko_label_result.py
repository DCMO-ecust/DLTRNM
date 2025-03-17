import numpy as np
import os
import csv
os.chdir('../data')
csv_file = '../merged_valid_ko_label_results.csv'
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['NO', 'TF', 'TG', 'Direction', 'Prediction', 'Correct'])
with open('processed_valid_ko_data.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)
ko_labels = np.array(data)
with open('model_ko_labels.csv', 'r') as f:
    for line in f:
        row = line.strip().split(',')
        if row[0] == 'NO':
            continue
        if row[1] in ko_labels and row[2] in ko_labels:
            if len(np.where(ko_labels == row[1])[0]) == 2:
                if len(np.where(ko_labels == row[2])[1]) == 2:
                    label = float(
                        ko_labels[int(np.where(ko_labels == row[1])[0][1])][int(np.where(ko_labels == row[2])[1][0])])
                else:
                    label = float(
                        ko_labels[int(np.where(ko_labels == row[1])[0][1])][int(np.where(ko_labels == row[2])[1])])
            else:
                continue
            if label == 0.0:
                continue
            if label == float(row[3]):
                flag = True
            else:
                flag = False
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([row[0], row[1], row[2], label, row[3], flag])
