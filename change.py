import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
# gene_classify = pd.read_csv('./data/gene_role.csv')
# cell = pd.read_csv('./data/uq1000_feature.csv')
# cell_head = cell.columns.tolist()
# gene_name = gene_classify['Gene'].tolist()
# same = 0
# for x in gene_name:
#     if x in cell_head:
#         same += 1
# print(same)
# gene_class = pd.read_csv('./used_gene_class.csv',sep='>')
# s = []
# gene = gene_class['path'].values.tolist()
# for i in range(len(gene)):
#     y = gene[i]
#     y = y.split(';')
#     s = s + y
#
#
# s1 = pd.read_csv('./s1.csv')
# s1 = s1['class_label'].values.tolist()
# ll = []
# for x in s1:
#     if(x not in s):
#         print(x)
#         ll.append(x)
#
# ll = pd.DataFrame({
#     'class_label': ll
# })
# ll.to_csv('./s1.csv')
#
#
# s = pd.read_csv('./data/uq1000_feature.csv')
# s1 = pd.read_csv('./data/Cell/cell_feature.csv')
# s2 = pd.read_csv('./data/Patient/patient_feature.csv')
# res = pd.read_csv('./data/Patient/Drug_patient_response.csv')
# s3 = res['Patient'].drop_duplicates()
# print('')

drug1 = pd.read_csv('./data/Cell/drugs_Smiles.csv')['Name'].tolist()
drug2 = pd.read_csv('./data/Patient/drugs_Smiles.csv')['Name'].tolist()

number = 0
for x in drug2:
    if(x.lower() not in drug1):
        print(x)
        number += 1
print(number)