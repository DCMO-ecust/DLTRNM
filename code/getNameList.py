import os
import numpy
from intermine.webservice import Service
import re
from urllib import request
import numpy as np

current_dir = os.getcwd()


def get_gene_name_from_expression_data():
    with open('pan_data.txt', 'r') as f:
        expression_gene_name_list = []
        for line in f:
            row = line.strip().split()
            if row[0][0] == 'Y':
                expression_gene_name_list.append(row[0])
            else:
                continue
        expression_gene_name_list = numpy.unique(expression_gene_name_list)
        expression_gene_name_list = [x.replace('.', '-') for x in expression_gene_name_list]
    return expression_gene_name_list


def get_gene_name_from_time_course_data():
    with open('time_course_data.txt', 'r') as f:
        time_course_gene_name_list = []
        for line in f:
            row = line.strip().split()
            if row[0] == 'GeneName':
                continue
            else:
                time_course_gene_name_list.append(row[0])
        time_course_gene_name_list = numpy.unique(time_course_gene_name_list)
        time_course_gene_name_list = [x.replace(',', '') for x in time_course_gene_name_list]
        return time_course_gene_name_list


def get_standard_and_systematic_name_from_sgd(name):
    standard_name = '0'
    systematic_name = '0'
    service = Service("https://yeastmine.yeastgenome.org/yeastmine/service")
    query = service.new_query("Gene")
    query.add_view(
        "primaryIdentifier", "proteins.symbol", "symbol", "secondaryIdentifier",
        "description"
    )
    query.add_constraint("organism.shortName", "=", "S. cerevisiae", code="B")
    query.add_constraint("Gene", "LOOKUP", name, code="A")
    for row in query.rows():
        if row["symbol"] is None and row["secondaryIdentifier"] == name:
            standard_name = row["secondaryIdentifier"]
            systematic_name = row["secondaryIdentifier"]
        elif row["symbol"] == name:
            standard_name = row["symbol"]
            systematic_name = row["secondaryIdentifier"]
    if standard_name == '0' and systematic_name == '0':
        for row in query.rows():
            if row["sgdAlias"] is None and row["secondaryIdentifier"] == name:
                standard_name = name
                systematic_name = row["secondaryIdentifier"]
            elif name in row["sgdAlias"]:
                standard_name = name
                systematic_name = row["secondaryIdentifier"]
            elif name not in row["sgdAlias"] and row["secondaryIdentifier"] == name:
                standard_name = name
                systematic_name = row["secondaryIdentifier"]
    return standard_name, systematic_name


def get_protein_name_and_systematic_name_from_yeastract(name):
    resp = request.urlopen('http://yeastract-plus.org/yeastract/scerevisiae/view.php?existing=protein&proteinname='
                           + name)
    content = resp.read().decode()
    protein_name = re.findall('Protein Name</th>\n    <td class="bgalign">(.+?)</td></tr>', content)
    if protein_name == []:
        protein_name = ['None']
    return protein_name


def get_expression_systematic_name_list():
    expression_systematic_name_list = []
    with open('pan_standard_and_systematic_name_list.txt', 'r') as f:
        for line in f:
            row = line.strip().split()
            if row[0] == 'standard_name':
                continue
            expression_systematic_name_list.append(row[1])
    return expression_systematic_name_list


def get_time_course_systematic_name_list():
    time_course_systematic_name_list = []
    # with open('time_course_standard_and_systematic_name_list.txt', 'r') as f:
    with open('time_course_standard_and_systematic_name_list.txt', 'r') as f:
        for line in f:
            row = line.strip().split()
            if row[0] == 'standard_name':
                continue
            time_course_systematic_name_list.append(row[1])
    # need to np.unique
    return time_course_systematic_name_list


def get_expression_strain_name():
    strains = []
    with open('pan_data.txt', 'r') as f:
        for line in f:
            row = line.strip().split()
            if row[1] == 'Strain':
                continue
            strains.append(row[1])
    strains = np.unique(strains)
    strains = strains.tolist()
    return strains


def get_time_course_strain_name():
    strains = []
    times = dict()
    with open('time_course_data.txt', 'r') as f:
        for line in f:
            row = line.strip().split()
            if row[2] == 'strain':
                continue
            strains.append(row[2])
    strains = np.unique(strains)
    strains = strains.tolist()
    for i in range(len(strains)):
        times[strains[i]] = []
    with open('time_course_data.txt', 'r') as f:
        for line in f:
            row = line.strip().split()
            if row[2] == 'strain':
                continue
            times[row[2]].append(float(row[3]))
    for i in range(len(strains)):
        times[strains[i]] = list(np.unique(sorted(times[strains[i]])))
    return strains, times


def get_interpolation_time_course_strain_name():
    strains = []
    times = dict()
    with open('interpolation_time_course_data.txt', 'r') as f:
        for line in f:
            row = line.strip().split()
            strains.append(row[1])
    strains = np.unique(strains)
    strains = strains.tolist()
    for i in range(len(strains)):
        times[strains[i]] = []
    with open('interpolation_time_course_data.txt', 'r') as f:
        for line in f:
            row = line.strip().split()
            times[row[1]].append(float(row[2]))
    for i in range(len(strains)):
        times[strains[i]] = list(np.unique(sorted(times[strains[i]])))
    return strains, times


def get_time_course_gene_dict():
    gene_dict = dict()
    with open('time_course_standard_and_systematic_name_list.txt', 'r') as f:
        for line in f:
            row = line.strip().split()
            if row[0] == 'none':
                gene_dict[row[1]] = row[1]
            else:
                gene_dict[row[0]] = row[1]
    return gene_dict


if __name__ == '__main__':
    '''os.chdir('../../data')
    a, b = get_time_course_strain_name()'''
