import torch
import torch.nn as nn
import os
import csv
from get_io import get_input_expression_list, get_time_course_single_net_input_data, get_input_time_course_list
from foundation_model import train_network, get_masks, weight_init, load_data, Net, CustomizedLinear, single_small_net, network_test
from getNameList import get_expression_systematic_name_list, get_time_course_systematic_name_list
from getdatasetdifference import get_tf_difference
from valid_custom_rule import valid_rules


if __name__ == '__main__':
    os.chdir('../data')
    torch.set_default_dtype(torch.float64)
    device = 'cuda'
    csv_file = 'valid_fine_tuned_output.py'
    # with open(csv_file, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['NO', 'name', 'excellent number', 'excellent rate', 'qualified number',
    #                      'qualified rate', 'fail number', 'fail rate'])
    expression_name_list = get_expression_systematic_name_list()
    time_course_name_list = get_time_course_systematic_name_list()
    common_genes = [g for g in expression_name_list if g in time_course_name_list]
    diff_tfs = get_tf_difference()
    name_list = common_genes + diff_tfs
    input_expression, input_num_exp = get_input_expression_list()  # 90s
    input_time_course, input_num_time = get_input_time_course_list()
    for i in range(0, len(name_list)):
        print('NO' + str(i) + ':validing %s network' % (name_list[i]))
        input_data, output_data, input_list = get_time_course_single_net_input_data(name_list[i], input_time_course,
                                                                                    input_expression)
        train_data, test_data = load_data(input_data, output_data, len(output_data), device)  # 5s
        pth = '../fine_tune_result/' + 'NO' + str(i) + '_' + str(name_list[i]) + '.pth'
        net = torch.load(pth)
        loss_func = nn.MSELoss(reduction='mean')
        for data in train_data:
            x, y = data
        result_1 = net(x).cpu().detach().tolist()
        result_1 = [item for sublist in result_1 for item in sublist]
        valid_1 = [item for sublist in y.cpu().detach().tolist() for item in sublist]
        flag_1 = valid_rules(result_1, valid_1)
        for data in test_data:
            x, y = data
        result_2 = net(x).cpu().detach().tolist()
        result_2 = [item for sublist in result_2 for item in sublist]
        valid_2 = [item for sublist in y.cpu().detach().tolist() for item in sublist]
        flag_2 = valid_rules(result_2, valid_2)
        flag = flag_1 + flag_2
        excellent = flag.count(2)
        qualified = flag.count(1)
        fail = flag.count(0)
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([str(i), name_list[i], excellent, excellent*100/len(flag), qualified,
                             qualified*100/len(flag), fail, fail*100/len(flag)])
        print('excellent: %d, %s%%' % (excellent, excellent*100/len(flag)))
        print('qualified: %d, %s%%' % (qualified, qualified*100/len(flag)))
        print('fail: %d, %s%%' % (fail, fail*100/len(flag)))

