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
    csv_file = '../model_ko_labels.csv'
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['NO', 'TF', 'TG', 'Direction'])
    expression_name_list = get_expression_systematic_name_list()
    time_course_name_list = get_time_course_systematic_name_list()
    common_genes = [g for g in expression_name_list if g in time_course_name_list]
    diff_tfs = get_tf_difference()
    name_list = common_genes + diff_tfs
    input_expression, input_num_exp = get_input_expression_list()  # 90s
    input_time_course, input_num_time = get_input_time_course_list()
    labels = []
    for i in range(len(name_list)):
        if name_list[i] == 'YIR017C':
            real_i = 5855
        elif name_list[i] == 'YKR034W':
            real_i = 5856
        elif name_list[i] == 'YLR013W':
            real_i = 5857
        elif name_list[i] == 'YIR013C':
            real_i = 5858
        else:
            real_i = i
        input_data, output_data, input_list = get_time_course_single_net_input_data(name_list[i], input_time_course,
                                                                                    input_expression)
        pth = '../fine_tune_result/' + 'NO' + str(real_i) + '_' + str(name_list[i]) + '.pth'
        net = torch.load(pth)
        net.eval()
        valid_input = torch.ones(1, len(input_list)).to(device)
        standard = net(valid_input)
        for j in range(len(input_list)):
            valid_input = torch.ones(1, len(input_list)).to(device)
            valid_input[0][j] = 0
            valid_output = net(valid_input)
            if valid_output-standard > 0:
                flag = 1
            elif valid_output - standard < 0:
                flag = -1
            else:
                flag = 0
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([real_i, input_list[j], name_list[i], flag])
