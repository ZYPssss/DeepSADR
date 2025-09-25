import pandas as pd
import math
import numpy as np
# threshold = 0.0
# #     reading files
# gdsc1_response = pd.read_csv('./data1/raw_dat/GDSC/GDSC1_fitted_dose_response_25Feb20.csv')
# gdsc2_response = pd.read_csv('./data1/raw_dat/GDSC/GDSC2_fitted_dose_response_25Feb20.csv')
# gex_features_df = pd.read_csv('./data1/preprocessed_dat/uq1000_feature.csv', index_col=0)
#
# #     selecting relevant columns
# gdsc1_sensitivity_df = gdsc1_response[['COSMIC_ID', 'DRUG_NAME', 'Z_SCORE']]
# gdsc2_sensitivity_df = gdsc2_response[['COSMIC_ID', 'DRUG_NAME', 'Z_SCORE']]
#
# gdsc1_sensitivity_df.loc[:, 'DRUG_NAME'] = gdsc1_sensitivity_df['DRUG_NAME'].str.lower()
# gdsc2_sensitivity_df.loc[:, 'DRUG_NAME'] = gdsc2_sensitivity_df['DRUG_NAME'].str.lower()
#
# #     mapping COSMIC ID to DepMap ID
# ccle_sample_info = pd.read_csv('./data1/raw_dat/CCLE/sample_info.csv', index_col=4)
# ccle_sample_info = ccle_sample_info.loc[ccle_sample_info.index.dropna()]
# ccle_sample_info.index = ccle_sample_info.index.astype('int')
#
# gdsc_sample_info = pd.read_csv('./data1/raw_dat/GDSC/gdsc_cell_line_annotation.csv', header=0, index_col=1)
# gdsc_sample_info = gdsc_sample_info.loc[gdsc_sample_info.index.dropna()]
# gdsc_sample_info.index = gdsc_sample_info.index.astype('int')
#
# gdsc_sample_mapping = gdsc_sample_info.merge(ccle_sample_info, left_index=True, right_index=True, how='inner')[
#     ['DepMap_ID']]
# gdsc_sample_mapping_dict = gdsc_sample_mapping.to_dict()['DepMap_ID']
#
# gdsc_sensitivity_df = pd.concat([gdsc1_sensitivity_df, gdsc2_sensitivity_df])
# gdsc_target_df = gdsc_sensitivity_df.groupby(['COSMIC_ID', 'DRUG_NAME']).mean()
# g = gdsc_target_df.reset_index().set_index('COSMIC_ID')
# g.index = g.index.map(gdsc_sample_mapping_dict)
# print(g.index.isnull().sum())
# g1 = g.reset_index()
# ass = g1.values.tolist()
# null_num = 0
# example = ass[57808]
# target_df = [['COSMIC_ID', 'DRUG_NAME', 'Z_SCORE']]
# for i in range(len(ass)):
#     if(isinstance(ass[i][0], str)):
#         target_df.append(ass[i])
#     else:
#         null_num += 1
# target_df = pd.DataFrame(target_df[1:], columns=target_df[0])
# target_df['drug_response'] = target_df['Z_SCORE'].apply(lambda x: 1 if x < threshold else 0)
# target_df.to_csv('./data/Cell/Drug_cell_response.csv',index=False)
# print(target_df)


# CCLE_Files = {
#     'gdsc1_sensitivity_df': gdsc1_sensitivity_df,
#     'gdsc2_sensitivity_df': gdsc2_sensitivity_df,
#     'gdsc_sample_mapping_dict': gdsc_sample_mapping_dict,
#     'gex_features_df': gex_features_df
# }
#
#
# drug = 'Bicalutamide'
# drug = drug.lower()
#
# gdsc1_sensitivity_df = CCLE_Files['gdsc1_sensitivity_df']
# gdsc2_sensitivity_df = CCLE_Files['gdsc2_sensitivity_df']
# gdsc_sample_mapping_dict = CCLE_Files['gdsc_sample_mapping_dict']
# gex_features_df = CCLE_Files['gex_features_df']
#
# #     getting dfs corresponding to a drug
# gdsc1_sensitivity_df = gdsc1_sensitivity_df.loc[gdsc1_sensitivity_df.DRUG_NAME.isin([drug])]
# gdsc2_sensitivity_df = gdsc2_sensitivity_df.loc[gdsc2_sensitivity_df.DRUG_NAME.isin([drug])]
#
# #     removing duplicate COSMIC IDs and taking mean of labels
# gdsc1_target_df = gdsc1_sensitivity_df.groupby(['COSMIC_ID', 'DRUG_NAME']).mean()
# gdsc2_target_df = gdsc2_sensitivity_df.groupby(['COSMIC_ID', 'DRUG_NAME']).mean()
#
#
# #     concat of 1 and 2
# gdsc1_target_df = gdsc1_target_df.loc[gdsc1_target_df.index.difference(gdsc2_target_df.index)]
# gdsc_target_df = pd.concat([gdsc1_target_df, gdsc2_target_df])
#
# #     mapping COSMIC ID to DepMap ID
# target_df = gdsc_target_df.reset_index().pivot_table(values=gdsc_target_df.columns, index='COSMIC_ID',
#                                                      columns='DRUG_NAME')
# target_df.index = target_df.index.map(gdsc_sample_mapping_dict)
# target_df = target_df.loc[target_df.index.dropna()]
#
# ########### Additional Changes ####################
# here = gdsc_target_df.columns.tolist()
# here.insert(0, 'COSMIC_ID')
# arr = np.concatenate((np.array(target_df.index).reshape(-1, 1),), axis=1)
# for metric in gdsc_target_df.columns:
#     arr = np.concatenate((arr, target_df[metric][drug].values.reshape(-1, 1)), axis=1)
# ccle_target_df = pd.DataFrame(arr, columns=here)
# ccle_target_df = ccle_target_df.set_index('COSMIC_ID')
# ccle_target_df.dropna(inplace=True)
# ########################################################################
# #     DepMap IDs of those samples: for which gex and labels are present
# #     so we have the gene expressions and all the labels for these samples in CCLE
# ccle_labeled_samples = gex_features_df.index.intersection(ccle_target_df.index)
#
# if threshold == None:
#     threshold = np.median(ccle_target_df['Z_SCORE'].loc[ccle_labeled_samples].values)
#
# ccle_labels = (ccle_target_df['Z_SCORE'].loc[ccle_labeled_samples] < threshold).astype('int')
# ccle_labeled_feature_df = gex_features_df.loc[ccle_labeled_samples]
# ccle_target_df['tZ_SCORE'] = ccle_labels
# assert all(ccle_labels.index == ccle_labeled_feature_df.index)
#
# unlabelled_data = ccle_labeled_feature_df
# labeled_data = ccle_target_df.loc[ccle_labeled_samples]
# labeled_data['drug_response'] = [{}] * len(labeled_data)
# diagnosis = False
# if diagnosis == True:
#     labeled_data['drug_response'] = labeled_data['tZ_SCORE'].apply(lambda x: {f"{drug} (Diagnosis)": x})
# else:
#     labeled_data['drug_response'] = labeled_data['tZ_SCORE'].apply(lambda x: {drug: x})


# gex_features_df = pd.read_csv('./data1/preprocessed_dat/uq1000_feature.csv', index_col=0)
#
# tcga_gex_feature_df = gex_features_df.loc[gex_features_df.index.str.startswith('TCGA')]
# tcga_gex_feature_df.index = tcga_gex_feature_df.index.map(lambda x: x[:12])
# tcga_gex_feature_df = tcga_gex_feature_df.groupby(level=0).mean()  # >>> unlabeled gexfile
#
#
# tcga_treatment_df = pd.read_csv('./data1/tcga/tcga_drug_first_treatment.csv')
# tcga_response_df = pd.read_csv('./data1/tcga/tcga_drug_first_response.csv')
#
# tcga_treatment_df.drop_duplicates(subset=['bcr_patient_barcode'], keep=False, inplace=True)  # >>> drug names
# tcga_treatment_df.set_index('bcr_patient_barcode', inplace=True)
#
# tcga_response_df.drop_duplicates(inplace=True)  # >>> shows tumour relapse time
# tcga_response_df.drop_duplicates(subset=['bcr_patient_barcode'], inplace=True)
# tcga_response_df.set_index('bcr_patient_barcode', inplace=True)
# drug = 'cis'
# drug_mapping_df = pd.read_csv('./data1/tcga_gdsc_drug_mapping.csv', index_col=0)
# if drug in ['tgem', 'tfu']:
#     gdsc_drug = drug_mapping_df.loc[drug[1:], 'gdsc_name']
#     drug_name = drug_mapping_df.loc[drug[1:], 'drug_name']
# else:
#     gdsc_drug = drug_mapping_df.loc[drug, 'gdsc_name']
#     drug_name = drug_mapping_df.loc[drug, 'drug_name']
#
# x = tcga_treatment_df['pharmaceutical_therapy_drug_name'].drop_duplicates()
# drug_list  = x.tolist()
# ass = [['Patient', 'DRUG_NAME', 'days_to_new_tumor_event_after_initial_treatment', 'response']]
# for drug_name in drug_list:
#     tcga_drug_barcodes = tcga_treatment_df.index[tcga_treatment_df['pharmaceutical_therapy_drug_name'] == drug_name]
#
#     drug_tcga_response_df = tcga_response_df.loc[tcga_drug_barcodes.intersection(tcga_response_df.index)]
#     labeled_tcga_gex_feature_df = tcga_gex_feature_df.loc[drug_tcga_response_df.index.intersection(tcga_gex_feature_df.index)]
#     labeled_df = tcga_response_df.loc[labeled_tcga_gex_feature_df.index]
#
#     days_threshold = np.median(labeled_df.days_to_new_tumor_event_after_initial_treatment)
#     g = labeled_df.reset_index()
#     g1 = g.values.tolist()
#     for i in range(len(g1)):
#         l = [g1[i][0], drug_name, g1[i][1]]
#         if(g1[i][1] > days_threshold):
#             l.append(1)
#         else:
#             l.append(0)
#         ass.append(l)
#
# drug_name = drug_mapping_df.loc['fu', 'drug_name']
#
# fu_res = pd.read_csv('./data1/preprocessed_dat/fu_res_gex.csv')
# index = fu_res.shape[0]
# num = 1
# for i in range(index):
#     fu_res.iloc[i, 0] = 'patient_{}'.format(num)
#     ass.append(['patient_{}'.format(num), drug_name, -1, 1])
#     num += 1
# fu_non = pd.read_csv('./data1/preprocessed_dat/fu_non_gex.csv')
# index = fu_non.shape[0]
# for i in range(index):
#     fu_non.iloc[i, 0] = 'patient_{}'.format(num)
#     ass.append(['patient_{}'.format(num), drug_name, -1, 0])
#     num += 1
#
# drug_name = drug_mapping_df.loc['gem', 'drug_name']
#
# gem_res = pd.read_csv('./data1/preprocessed_dat/gem_res_gex.csv')
# index = gem_res.shape[0]
# for i in range(index):
#     gem_res.iloc[i, 0] = 'patient_{}'.format(num)
#     ass.append(['patient_{}'.format(num), drug_name, -1, 1])
#     num += 1
# gem_non = pd.read_csv('./data1/preprocessed_dat/gem_non_gex.csv')
# index = gem_non.shape[0]
# for i in range(index):
#     gem_non.iloc[i, 0] = 'patient_{}'.format(num)
#     ass.append(['patient_{}'.format(num), drug_name, -1, 0])
#     num += 1
# y = tcga_gex_feature_df.reset_index()
# x = pd.concat([y, fu_res, fu_non, gem_res, gem_non])
#
#
#
# ASS = pd.DataFrame(ass[1:], columns=ass[0])
# ASS.to_csv('./data/Patient/Drug_patient_response.csv', index=False)
# x.to_csv('./data/Patient/patient_feature.csv', index=False)


gex_features_df = pd.read_csv('./data1/preprocessed_dat/uq1000_feature.csv', index_col=0)

tcga_gex_feature_df = gex_features_df.loc[gex_features_df.index.str.startswith('ACH')]
x = tcga_gex_feature_df.reset_index()
x.to_csv('./data/Cell/cell_feature.csv',index=False)

print(tcga_gex_feature_df)

