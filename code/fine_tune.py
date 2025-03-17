import torch
import torch.nn as nn
import os
import math
import numpy as np
import pandas as pd
import sys
import time
from get_io import get_input_expression_list, get_time_course_single_net_input_data, get_input_time_course_list
from foundation_model import train_network, get_masks, weight_init, load_data, Net, CustomizedLinear, single_small_net, network_test
from getNameList import get_expression_systematic_name_list, get_time_course_systematic_name_list
from getdatasetdifference import get_tf_difference


def reconstruction_small_net(i, name_list, input_list, input_expression, device, expression_name_list):
    if name_list[i] in expression_name_list:
        pth = '../result/' + 'NO' + str(expression_name_list.index(name_list[i])) \
              + '_' + str(name_list[i]) + '.pth'
        original_model = torch.load(pth)
        if input_list == input_expression[name_list[i]]:
            new_model = single_small_net(len(input_list), len(input_list) * 3, 1).to(device)
        elif input_list != input_expression[name_list[i]]:
            new_model = single_small_net(len(input_list), len(input_expression[name_list[i]]) * 3, 1).to(device)
        new_model = weight_init(new_model, nn.Linear, 'sigmoid')
        new_model.hidden2.weight.data = original_model.hidden2.weight.data.clone()
        new_model.hidden2.bias.data = original_model.hidden2.bias.djata.clone()
        new_model.zero_grad()
        new_model.hidden2.weight.requires_grad = False
        new_model.hidden2.bias.requires_grad = False
    elif name_list[i] not in expression_name_list:
        new_model = single_small_net(len(input_list), len(input_list) * 3, 1).to(device)
        new_model = weight_init(new_model, nn.Linear, 'sigmoid')
        new_model.zero_grad()
    return new_model


if __name__ == '__main__':
    os.chdir('../data')
    torch.set_default_dtype(torch.float64)
    device = 'cuda'
    expression_name_list = get_expression_systematic_name_list()
    time_course_name_list = get_time_course_systematic_name_list()
    common_genes = [g for g in expression_name_list if g in time_course_name_list]
    diff_tfs = get_tf_difference()
    name_list = common_genes + diff_tfs
    input_expression, input_num_exp = get_input_expression_list()  # 90s
    input_time_course, input_num_time = get_input_time_course_list()
    final_training_loss = []
    final_testing_loss = []
    each_gene_tf_count = pd.DataFrame()
    each_gene_tf_count['name'] = name_list
    counts = []
    for i in range(len(name_list)):
        start_time = time.time()
        input_data, output_data, input_list = get_time_course_single_net_input_data(name_list[i], input_time_course,
                                                                                    input_expression)
        counts.append(len(input_list))
        if i < len(name_list)-1:
            continue
        each_gene_tf_count['counts'] = counts
        each_gene_tf_count.to_csv('each_gene_tf_count.csv', index=False)
        print(1)
        train_data, test_data = load_data(input_data, output_data, len(output_data), device)  # 5s
        max_iter = 200*math.floor(len(output_data)*0.9)
        if max_iter > 20000:
            max_iter = 20000
        print('training NO:%d %s' % (i, name_list[i]))
        if len(output_data) <= 20:
            train_times = 5
        elif 20 < len(output_data) <= 30:
            train_times = 4
        elif 30 < len(output_data) <= 40:
            train_times = 3
        elif 40 < len(output_data) <= 50:
            train_times = 2
        if len(output_data) > 50:
            new_model = reconstruction_small_net(i, name_list, input_list, input_expression, device,
                                                 expression_name_list)
            train_loss = train_network(new_model, 0.001, max_iter, train_data, test_data)
            final_training_loss.append(train_loss)
            test_loss = network_test(new_model, test_data)
            final_testing_loss.append(test_loss)
        elif len(output_data) <= 50:
            print('Need to train %d times' % train_times)
            model_state = []
            train_losses = []
            test_losses = []
            for p in range(train_times):
                print('training NO%d time' % (p+1))
                train_data, test_data = load_data(input_data, output_data, len(output_data), device)  # 5s
                new_model = reconstruction_small_net(i, name_list, input_list, input_expression, device,
                                                     expression_name_list)
                train_loss = train_network(new_model, 0.001, max_iter, train_data, test_data)
                model_state.append(new_model.state_dict())
                test_loss = network_test(new_model, test_data)
                train_losses.append(float(train_loss.detach().cpu()))
                test_losses.append(float(test_loss.detach().cpu()))
            min_params = model_state[test_losses.index(min(test_losses))]
            new_model = reconstruction_small_net(i, name_list, input_list, input_expression, device,
                                                 expression_name_list)
            new_model.load_state_dict(min_params)
            print('mean train loss:%f' % (np.mean(train_losses)))
            print('mean test loss:%f' % (np.mean(test_losses)))
            print('final testing error:%f' % (min(test_losses)))
            final_training_loss.append(train_losses[test_losses.index(min(test_losses))])
            final_testing_loss.append(min(test_losses))
        pth = '../fine_tune_result/' + 'NO' + str(i) + '_' + str(name_list[i]) + '.pth'
        # torch.save(new_model, pth)
        end_time = time.time()
        print('NO' + str(i) + ' using time:' + str(end_time-start_time))
    # with open('fine_tune_loss.txt', 'a') as f:
    #     for k in range(len(final_testing_loss)):
    #         f.writelines(str(final_training_loss[k]) + '\t' + str(final_testing_loss[k]) + '\n')
