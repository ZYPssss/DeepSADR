# DeepSADR
=======
# DeepSADR
DeepSADR: A Deep Learning Model Based on Subsequence Interaction and
Adaptive Readout for Enhanced Drug Response Prediction in Cancer Patients


## Overview 
We propose DeepSADR, a transfer learning model for
drug response prediction from cell lines to patients, built on
subsequence interaction and adaptive readout. DeepSADR
adopts a ’pre-training + fine-tuning’ strategy. We construct
sub-sequence interaction graphs to explore the associations
between drug and gene subsequences, thereby improving
model performance and interpretability. To achieve effective
model transfer in the drug response(domain), we introduce
an adaptive readout function to learn domain-invariant drug
response features, thereby improving the model’s predictive
performance on patient data.

## Data
'data/Cell : Includes cell lines geneomic profiles data, drug Smiles sequences, and drug response data.


'data/Patient': Includes patients geneomic profiles data, drug Smiles sequences, and drug response data.


'data /split_cell_lines.csv': Classification results for genes



## Environment
`You can create a conda environment for DeepSADR  by ‘conda env create -f environment.yml‘.`


## Train and test
- ### step 1
  - #### pre-trian 
        'python pretrain_model.py'
- ### step 2 (regression experiment):
  - #### fine-tune 
        'python fine_tune_model.py'
 
