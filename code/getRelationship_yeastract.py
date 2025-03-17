import os
from urllib import request
import re
import numpy as np
from getNameList import get_expression_systematic_name_list
from getNameList import get_time_course_systematic_name_list


def find_tfs(tf_name):
    resp = request.urlopen('http://yeastract-plus.org/yeastract/scerevisiae/view.php?existing=protein&proteinname='
                           + tf_name)
    content = resp.read().decode()
    protein_name = re.findall('Protein Name</th>\n    <td class="bgalign">(.+?)</td></tr>', content)
    if protein_name == []:
        protein_name = ['None']
    if re.findall('Documented targets', content):
        return tf_name, protein_name
    else:
        return 0, 0


def get_relationship_from_downloaded_files(file_name, tf_name_file, systematic_name_file, output_file_name):
    with open(file_name, 'r') as f1:
        relation = dict()
        target = []
        tf = ''
        for line1 in f1:
            row1 = line1.strip().split(';')
            if row1[0] != '' and row1[0] != 'TF':
                relation[tf] = target
                target = []
                tf_name = row1[0]
                with open(tf_name_file, 'r') as f2:
                    for line2 in f2:
                        row2 = line2.strip().split()
                        if row2[1] == tf_name:
                            tf = row2[0]
                with open(systematic_name_file, 'r') as f3:
                    for line3 in f3:
                        row3 = line3.strip().split()
                        if row3[0] == row1[1]:
                            target.append(row3[1])
            else:
                with open(systematic_name_file, 'r') as f3:
                    for line3 in f3:
                        row3 = line3.strip().split()
                        if row3[0] == row1[1]:
                            target.append(row3[1])
            relation[tf] = target
        with open(output_file_name, 'a') as f:
            for i in range(1, len(relation)):
                f.writelines(list(relation)[i] + '\t'
                             + ','.join(relation.get(list(relation)[i])) + '\n')
    return relation


if __name__ == '__main__':
    os.chdir('../../data')
    # relation = get_relationship_from_downloaded_files('fixed_time_course_relationship_111-225.csv', 'fixed_time_tfs.txt',
    #                                                   'fixed_time_names.txt', 'fixed_time_relationships.txt')

