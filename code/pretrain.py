import torch
import os
import torch.nn as nn
import sys
import time
from get_io import get_expression_input_data, get_input_expression_list, get_expression_single_net_input_data
from foundation_model import train_network, get_masks, weight_init, load_data, Net, CustomizedLinear, single_small_net
from getNameList import get_expression_systematic_name_list


if __name__ == '__main__':
    os.chdir('../data')
    torch.set_default_dtype(torch.float64)
    device = 'cuda'
    # device = 'cpu'
    name_list = get_expression_systematic_name_list()
    input, input_num = get_input_expression_list() #90s
    for i in range(len(name_list)):
        start_time = time.time()
        print('NO' + str(i) + ':training %s network' % (name_list[i]))
        x, y = get_expression_single_net_input_data(name_list[i], input) #15s
        dl = load_data(x, y, len(y), device) #5s
        net = single_small_net(len(input[name_list[i]]), len(input[name_list[i]])*3, 1).to(device)
        net = weight_init(net, nn.Linear, 'sigmoid')
        net.zero_grad()
        train_network(net, 0.005, 2500, dl)
        pth = '../result/' + 'NO' + str(i) + '_' + str(name_list[i]) + '.pth'
        # torch.save(net, pth)
        end_time = time.time()
        print('NO' + str(i) + ' using time:' + str(end_time-start_time))
    breakpoint()