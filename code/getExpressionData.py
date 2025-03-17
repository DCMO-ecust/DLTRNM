import os
import csv

current_dir = os.getcwd()


def read_pan_data():
    with open('final_data_annotated_merged_04052022.tab', 'r', encoding='utf-8') as f:
        expression_data = []
        for line in f:
            row = line.strip().split(',')
            if row[-1] == 'absent':
                continue
            else:
                expression_data.append([row[-5], row[-4], row[-2], row[-1]])
    return pan_data


def read_time_course_data():
    with open('idea_expression_data.tsv', 'r') as f:
        time_course_data = []
        for line in f:
            row = line.strip().split()
            if row[6] == "IMP2'":
                row[6] = 'IMP21'
            time_course_data.append([row[6], row[0], row[1], row[5], row[14]])
    return time_course_data


if __name__ == '__main__':
    '''os.chdir('../../data/original data')
    expression_data = read_pan_data()
    time_course_data = read_time_course_data()
    os.chdir('../')
    with open('expression_data.txt', 'w', newline='') as f:
        for i in range(len(expression_data)):
            f.writelines(expression_data[i][0] + '\t' + expression_data[i][1] + '\t' + expression_data[i][2] + '\t'
                         + expression_data[i][3] + '\n')
    with open('time_course_data.txt', 'w', newline='') as f:
        for i in range(len(time_course_data)):
            f.writelines(time_course_data[i][0] + '\t' + time_course_data[i][1] + '\t' + time_course_data[i][2] + '\t'
                         + time_course_data[i][3] + '\t' + time_course_data[i][4] + '\n')'''
