import os
import math
import random
import sys
import numpy as np
from data_processing import expression_data_tpm_log2, time_course_reverse_log
from getNameList import get_expression_strain_name, get_expression_systematic_name_list, get_time_course_strain_name, get_time_course_systematic_name_list, get_time_course_gene_dict, get_interpolation_time_course_strain_name


def get_input_time_course_list():
    input = dict()
    input_nums = []
    with open('time_course_standard_and_systematic_name_list.txt', 'r') as f:
        for line in f:
            row = line.strip().split()
            if row[0] == 'standard_name':
                continue
            with open('time_course_relationships.txt', 'r') as file:
                tfs = []
                for lines in file:
                    rows = lines.strip().split()
                    if row[1] in rows[1].split(','):
                        tfs.append(rows[0])
                if not tfs:
                    tfs.append(row[1])
            input[row[1]] = tfs
            input_nums.append(len(tfs))
    return input, input_nums


def get_input_expression_list():
    input = dict()
    input_nums = []
    with open('pan_standard_and_systematic_name_list.txt', 'r') as f:
        for line in f:
            row = line.strip().split()
            if row[0] == 'standard_name':
                continue
            with open('pan_relationships.txt', 'r') as file:
                tfs = []
                for lines in file:
                    rows = lines.strip().split()
                    if row[1] in rows[1].split(','):
                        tfs.append(rows[0])
                if not tfs:
                    tfs.append(row[1])
            input[row[1]] = tfs
            input_nums.append(len(tfs))
    return input, input_nums


def get_expression_input_data():
    name_list = get_expression_systematic_name_list()
    strain_list = get_expression_strain_name()
    name_num = len(name_list)
    strain_num = len(strain_list)
    output_data = np.zeros((strain_num, name_num))
    with open('pan_data.txt', 'r') as f:
        for line in f:
            row = line.strip().split()
            if row[0] in name_list and row[1] in strain_list:
                output_data[strain_list.index(row[1])][name_list.index(row[0])] = expression_data_tpm_log2(float(row[2]))
    trans = np.transpose(output_data).tolist()
    input, input_nums = get_input_expression_list()
    input_data = []
    for i in range(name_num):
        tgs = input[name_list[i]]
        for j in range(len(tgs)):
            input_data.append(trans[name_list.index(tgs[j])])
    input_data = np.transpose(input_data).tolist()
    output_data = output_data.tolist()
    return input_data, output_data


def get_expression_single_net_input_data(tg_name, input):
    strain_list = get_expression_strain_name()
    strain_num = len(strain_list)
    tf_num = len(input[tg_name])
    input_data = np.zeros((strain_num+1, tf_num))
    output_data = np.zeros((strain_num+1, 1))
    for i in range(tf_num):
        input_data[strain_num, i] = 1.
    output_data[strain_num, 0] = 1.
    with open('pan_data.txt', 'r') as f:
        for line in f:
            row = line.strip().split()
            row[0] = row[0].replace('.', '-')
            if row[0] in input[tg_name] and row[1] in strain_list:
                input_data[strain_list.index(row[1])][input[tg_name].index(row[0])] = expression_data_tpm_log2(float(row[2]))
            if row[0] == tg_name and row[1] in strain_list:
                output_data[strain_list.index(row[1])][0] = expression_data_tpm_log2(float(row[2]))
    input_data = input_data.tolist()
    output_data = output_data.tolist()
    return input_data, output_data


def get_time_course_single_net_input_data(tg_name, input_time_course, input_expression):
    strain_list, time_list = get_interpolation_time_course_strain_name()
    strain_num = len(strain_list)
    total_num = 0
    input_index_name = []
    output_index_name = []
    for i in range(strain_num):
        total_num = total_num + len(time_list[strain_list[i]])-1
        for j in range(len(time_list[strain_list[i]])-1):
            input_index_name.append(strain_list[i]+'_'+'input'+'_'+str(float(time_list[strain_list[i]][j])))
        for j in range(1, len(time_list[strain_list[i]])):
            output_index_name.append(strain_list[i] + '_' + 'output' + '_' + str(float(time_list[strain_list[i]][j])))
    if tg_name in input_expression:
        set1 = set(input_time_course[tg_name])
        set2 = set(input_expression[tg_name])
        diff1 = list(set1-set2)#in time_course but not in expression
        input_list = input_expression[tg_name] + diff1
    elif tg_name not in input_expression:
        input_list = input_time_course[tg_name]
    tf_num = len(input_list)
    input_data = np.ones((total_num+1, tf_num))
    output_data = np.ones((total_num+1, 1))
    with open('interpolation_time_course_data.txt', 'r') as f:
        for line in f:
            row = line.strip().split()
            if row[0] == 'GeneName':
                continue
            if row[0] in input_list and float(row[2]) != time_list[row[1]][-1]:
                input_data[input_index_name.index(row[1]+'_'+'input'+'_'+str(float(row[2])))][input_list.index(row[0])]\
                    = time_course_reverse_log(float(row[3]))
            elif row[0] == tg_name and float(row[2]) != time_list[row[1]][0]:
                output_data[output_index_name.index(row[1] + '_' + 'output' + '_' + str(float(row[2])))][0]\
                    = time_course_reverse_log(float(row[3]))
    bad_input = []
    bad_output = []
    for i in range(len(input_data)-2, -1, -1):
        flag = 0
        for j in range(len(input_data[i])):
            if input_data[i][j] != 1. and output_data[i] != 1.:
                flag = 1
            elif (input_data[i][j] >= 1.2 or input_data[i][j] <= 0.8) and output_data[i] == 1.:
                flag = 2
        if flag == 2:
            bad_input.append(list(input_data[i]))
            bad_output.append(list(output_data[i]))
            input_data = np.delete(input_data, i, axis=0)
            output_data = np.delete(output_data, i, axis=0)
        elif flag == 0:
            input_data = np.delete(input_data, i, axis=0)
            output_data = np.delete(output_data, i, axis=0)
    bad_input = np.array(bad_input)
    bad_output = np.array(bad_output)
    number = math.ceil(len(output_data)*0.2)
    if len(bad_output) <= number:
        input_data = np.vstack((input_data, bad_input))
        output_data = np.vstack((output_data, bad_output))
    elif len(bad_output) > number:
        chosen = random.sample(range(len(bad_output)), number)
        for j in chosen:
            input_data = np.vstack((input_data, bad_input[j]))
            output_data = np.vstack((output_data, bad_output[j]))
    return input_data, output_data, input_list


if __name__ == '__main__':
    '''os.chdir('../../data')
    input_time_course, time_course_nums = get_input_time_course_list()
    input_expression, expression_nums = get_input_expression_list()
    input_data, output_data, input_list = get_time_course_single_net_input_data('YKR034W', input_time_course, input_expression)
    print(1)'''
