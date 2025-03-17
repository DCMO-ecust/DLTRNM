import numpy as np
import os
from getNameList import get_expression_systematic_name_list, get_time_course_systematic_name_list


def get_tf_difference():
    expression_tfs = []
    time_course_tfs = []
    with open('pan_tfs.txt', 'r') as f:
        for line in f:
            row = line.strip().split()
            expression_tfs.append(row[0])
    with open('time_course_tfs.txt', 'r') as f:
        for line in f:
            row = line.strip().split()
            time_course_tfs.append(row[0])
    set1 = set(expression_tfs)
    set2 = set(time_course_tfs)
    diff1 = list(set1 - set2)
    diff2 = list(set2 - set1)
    diff = diff1 + diff2
    return diff


def get_gene_difference():
    expression_gene = get_expression_systematic_name_list()
    time_course_gene = get_time_course_systematic_name_list()
    set1 = set(expression_gene)
    set2 = set(time_course_gene)
    diff1 = list(set1 - set2)
    diff2 = list(set2 - set1)
    diff = diff1 + diff2
    return diff


if __name__ == '__main__':
    '''os.chdir('../../data')
    get_gene_difference()
    get_tf_difference()'''
