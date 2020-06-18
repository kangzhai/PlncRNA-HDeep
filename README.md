# PlncRNA-HDeep
The related datasets and scoure codes of PlncRNA-HDeep privided by Q. Kang

The latest version is updated on 2020.06.18

# Introduction
PlncRNA-HDeep is a method using hybrid deep leaning based on two encoding styles for plant lncRNA prediction. It is implemented by Keras and all main scripts are written by Python on PC under a Microsoft Windows 10 operating system.

The repository can be downloaded locally by clicking "clone or download" button. PlncRNA-HDeep can be applied directly without installation.

# Dependency
windows operating system

python 3.6.5

Keras 2.2.4

# Datasets
The datasets can be obtained by unzipping "Datasets.zip".

"PositiveDataset.fasta" includes the names and sequences of 18000 lncRNAs.

"NegativeDataset.fasta" includes the names and sequences of 18000 mRNAs.

"TotalDataset.fasta" includes the informantion and labels of all positive and negative samples.

# Data description
"Data description.xlsx" lists the information of the used databases in the paper.

# Extracted Features
"Extracted features.xlsx" lists the information of the extracted features for shallow machine learning training in the paper.

# Usage
Open the console or powershell in the local folder and copy the following command to run PlncRNA-HDeep.

command: python PlncRNA-HDeep.py

This python script can also be directly run using python IDE such as pyCharm.

Explanation: This is a script that repeats the experiment in the paper. This script includes two encoding styles and three hybrid strategies descripted in the paper to provide references for other related research. The input of the method are the sequences and labels of the samples. The output are test results, including true positive (TP), false positive (FP), true negative (TN), false negative(FN), sensitivity (TPR), specificity (TNR), preision (PPV), negative predictive value (NPV), false negative rate (FNR), false positive rate (FPR), false discovery rate (FDR), false omission rate (FOR), accuracy (ACC), F1 score (F1), matthews correlation coefficient (MCC), bookmaker informedness (BM) and Markedness (MK).

We will upload the trained models as soon as possible in the next version, and provide the codes and commands for users to directly predict lncRNA.

# Reference
Wait for updating
