import os
from intermine.webservice import Service
import csv


def get_query():
    csv_file = '../label_from_sgd.csv'
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Regulator Standard Name', 'Regulator Systematic Name', 'Gene Standard Name',
                         'Gene Systematic Name', 'Experiment Condition', 'Strain Background', 'Regulation Direction',
                         'Regulation Type', 'Regulator Type', 'Datasource', 'Annotation Type'])
    service = Service("https://yeastmine.yeastgenome.org/yeastmine/service")
    query = service.new_query("Gene")
    query.add_constraint("regulatoryRegions", "TFBindingSite")
    query.add_view(
        "regulatoryRegions.regulator.symbol",
        "regulatoryRegions.regulator.secondaryIdentifier", "symbol",
        "secondaryIdentifier", "regulatoryRegions.experimentCondition",
        "regulatoryRegions.strainBackground",
        "regulatoryRegions.regulationDirection", "regulatoryRegions.regulationType",
        "regulatoryRegions.regulatorType", "regulatoryRegions.datasource",
        "regulatoryRegions.annotationType"
    )
    query.add_constraint("regulatoryRegions.regulationDirection", "IS NOT NULL", code="C")
    query.add_constraint("regulatoryRegions.regulatorType", "=", "transcription factor", code="A")
    query.add_constraint("regulatoryRegions.regulationType", "=", "transcription", code="B")
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        for row in query.rows():
            writer.writerow([row["regulatoryRegions.regulator.symbol"],
                             row["regulatoryRegions.regulator.secondaryIdentifier"], row["symbol"],
                             row["secondaryIdentifier"], row["regulatoryRegions.experimentCondition"],
                             row["regulatoryRegions.strainBackground"], row["regulatoryRegions.regulationDirection"],
                             row["regulatoryRegions.regulationType"], row["regulatoryRegions.regulatorType"],
                             row["regulatoryRegions.datasource"], row["regulatoryRegions.annotationType"]])


if __name__ == '__main__':
    get_query()

