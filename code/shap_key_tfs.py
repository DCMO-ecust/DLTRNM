import torch
import torch.nn as nn
import os
import csv
from get_io import get_input_expression_list, get_time_course_single_net_input_data, get_input_time_course_list
from getNameList import get_expression_systematic_name_list, get_time_course_systematic_name_list
from getdatasetdifference import get_tf_difference
import numpy as np
import shap
import time
import pandas as pd
from scipy.stats import pearsonr


if __name__ == '__main__':
    os.chdir('../data')
    torch.set_default_dtype(torch.float64)
    device = 'cuda'
    csv_file = '../shap_with_analysis.csv'
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['NO', 'TG', 'key_TFs', 'mean_shap', 'std_shap', 'CV'])
    expression_name_list = get_expression_systematic_name_list()
    time_course_name_list = get_time_course_systematic_name_list()
    common_genes = [g for g in expression_name_list if g in time_course_name_list]
    diff_tfs = get_tf_difference()
    name_list = common_genes + diff_tfs
    input_expression, input_num_exp = get_input_expression_list()  # 90s
    input_time_course, input_num_time = get_input_time_course_list()
    for i in range(len(name_list)):
        if name_list[i] == 'YIR017C':
            real_j = 5855
        elif name_list[i] == 'YKR034W':
            real_j = 5856
        elif name_list[i] == 'YLR013W':
            real_j = 5857
        elif name_list[i] == 'YIR013C':
            real_j = 5858
        else:
            real_j = i
        print('NO' + str(real_j) + ':searching %s network' % (name_list[i]))
        input_data, output_data, input_list = get_time_course_single_net_input_data(name_list[i], input_time_course,
                                                                                    input_expression)
        pth = '../fine_tune_result/' + 'NO' + str(real_j) + '_' + str(name_list[i]) + '.pth'
        net = torch.load(pth)
        net.eval()
        start_time = time.time()
        input_tensor = torch.tensor(input_data, dtype=torch.float64, device=device)
        explainer = shap.DeepExplainer(net, input_tensor)
        shap_values = explainer.shap_values(input_tensor)
        shap_values = np.array(shap_values)
        mean_shap_values = np.mean(np.abs(shap_values), axis=0)
        end_time = time.time()
        df = pd.DataFrame({
            'Feature': input_list,
            'SHAP Value': mean_shap_values
        })
        df_sorted = df.sort_values(by='SHAP Value', ascending=False)
        mean_values = df['SHAP Value'].mean()
        std_values = df['SHAP Value'].std()
        CV = std_values/mean_values
        top_features = df_sorted.head(5)
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([str(real_j), name_list[i], top_features['Feature'].values, mean_values, std_values, CV])
        # correlation_coefficients = []
        # p_values = []
        # for j in range(len(input_list)):
        #     correlation_coefficient, p_value = pearsonr(input_data.T[j][0:52], output_data.flatten()[0:52])
        #     correlation_coefficients.append(correlation_coefficient)
        #     p_values.append(p_value)
        # df['correlation_coefficients'] = correlation_coefficients
        # df['p_values'] = p_values