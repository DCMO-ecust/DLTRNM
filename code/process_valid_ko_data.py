import os
import numpy as np

if __name__ == '__main__':
    os.chdir('../data')
    with open('time_course_standard_and_systematic_name_list.txt', 'r') as f:
        modeling_gene_names = dict()
        for line in f:
            row = line.strip().split()
            modeling_gene_names[row[0]] = row[1]
    with open('time_course_tfs.txt', 'r') as f:
        modeling_tf_names = []
        for line in f:
            row = line.strip().split()
            modeling_tf_names.append(row[0])
    with open('SuppTable1A_KOTargetGenes_Matrix_DixExp_p0.05.txt', 'r') as f:
        TF_name_list = []
        gene_name_list = []
        for line in f:
            row = line.strip().split()
            if len(row[1]) >= 7:
                data = np.zeros((1, len(row)))
                for i in range(len(row)):
                    gene_name_list.append(row[i])
            else:
                if row[0].upper() in modeling_gene_names.keys():
                    if modeling_gene_names[row[0].upper()] in modeling_tf_names:
                        TF_name_list.append([modeling_gene_names[row[0].upper()]])
                        data_tmp = np.zeros((1, len(row) - 1))
                        for i in range(1, len(row)):
                            data_tmp[0][i-1] = row[i]
                        data = np.vstack((data, data_tmp))
        data = data[1:]
    print(1)
