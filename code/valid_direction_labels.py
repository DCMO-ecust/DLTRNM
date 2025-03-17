import torch
import torch.nn as nn
import os
import csv
from get_io import get_input_expression_list, get_time_course_single_net_input_data, get_input_time_course_list
from getNameList import get_expression_systematic_name_list, get_time_course_systematic_name_list
from getdatasetdifference import get_tf_difference


if __name__ == '__main__':
    os.chdir('../data')
    torch.set_default_dtype(torch.float64)
    device = 'cuda'
    csv_file = '../valid_direction_labels_down.csv'
    # csv_file = '../valid_direction_labels_up.csv'
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['NO', 'TF', 'TG', 'Direction', 'Prediction', 'Correct'])
    expression_name_list = get_expression_systematic_name_list()
    time_course_name_list = get_time_course_systematic_name_list()
    common_genes = [g for g in expression_name_list if g in time_course_name_list]
    diff_tfs = get_tf_difference()
    name_list = common_genes + diff_tfs
    input_expression, input_num_exp = get_input_expression_list()  # 90s
    input_time_course, input_num_time = get_input_time_course_list()
    labels = []
    with open('../label_from_sgd.csv', mode='r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            labels.append(row)
    for i in range(len(labels)):
        if labels[i][3] not in name_list:
            continue
        elif labels[i][3] in name_list:
            j = name_list.index(labels[i][3])
            if labels[i][3] == 'YIR017C':
                real_j = 5855
            elif labels[i][3] == 'YKR034W':
                real_j = 5856
            elif labels[i][3] == 'YLR013W':
                real_j = 5857
            elif labels[i][3] == 'YIR013C':
                real_j = 5858
            else:
                real_j = j
            print('NO' + str(real_j) + ':validing %s network' % (name_list[j]))
            input_data, output_data, input_list = get_time_course_single_net_input_data(name_list[j], input_time_course,
                                                                                    input_expression)
            if labels[i][1] not in input_list:
                print('Not an available label')
                continue
            elif labels[i][1] in input_list:
                k = input_list.index(labels[i][1])
                pth = '../fine_tune_result/' + 'NO' + str(real_j) + '_' + str(name_list[j]) + '.pth'
                net = torch.load(pth)
                net.eval()
                valid_input = torch.ones(1, len(input_list)).to(device)
                standard = net(valid_input)
                flag = 1
                if labels[i][6] == 'positive':
                    for m in range(2, 6):
                        valid_input[0][k] = 1/m
                        valid_output = net(valid_input)
                        if valid_output > standard:
                            flag = 0
                if labels[i][6] == 'negative':
                    for m in range(2, 6):
                        valid_input[0][k] = 1/m
                        valid_output = net(valid_input)
                        if valid_output < standard:
                            flag = 0
            if flag == 0:
                result = False
                prediction = 'other'
            elif flag == 1:
                result = True
                prediction = labels[i][6]
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([j, labels[i][1], labels[i][3], labels[i][6], prediction, result])
print(1)
