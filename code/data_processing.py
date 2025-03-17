import math
import os
from scipy.interpolate import Rbf
from getNameList import get_time_course_strain_name, get_time_course_systematic_name_list, get_time_course_gene_dict


def expression_data_tpm_log2(tpm):
    log2tpm = math.log(tpm+1, 2)
    return log2tpm


def time_course_reverse_log(log2data):
    reverse_log = math.log(2**log2data + 1, 2)
    # reverse_log = 2**log2data
    return reverse_log


def get_interpolation_time_course_data_file():
    target_times = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    strains, times = get_time_course_strain_name()
    gene_names = get_time_course_systematic_name_list()
    gene_dict = get_time_course_gene_dict()
    index_list = []
    for i in range(len(gene_names)):
        for j in range(len(strains)):
            index_list.append(gene_names[i]+'_'+strains[j])
    origin_data = dict()
    for k in range(len(index_list)):
        origin_data[index_list[k]] = [0] * len(times[index_list[k].split('_')[-1]])
    with open('time_course_data.txt', 'r') as f:
        for line in f:
            row = line.strip().split()
            if row[0] == 'GeneName':
                continue
            row[0] = gene_dict[row[0].replace(',', '')]
            origin_data[row[0] + '_' + row[2]][times[row[2]].index(float(row[3]))] = float(row[4])
    for k in range(len(index_list)):
        y1 = origin_data[index_list[k]].copy()
        x1 = times[index_list[k].split('_')[-1]].copy()
        rbf = Rbf(x1, y1, function='gaussian')
        interpolation = rbf(target_times)
        for i in range(len(target_times)):
            with open('interpolation_time_course_data.txt', 'a') as f:
                f.writelines(index_list[k].split('_')[0] + '\t' + index_list[k].split('_')[-1] + '\t'
                             + str(target_times[i]) + '\t' + str(interpolation[i]) + '\n')


if __name__ == '__main__':
    '''os.chdir('../data')'''
