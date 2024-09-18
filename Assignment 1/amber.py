import os
import pandas as pd

# quick solve to directory issue
os.chdir(r'C:\Users\Gebruiker\Data-Mining-1\Assignment 1')

from final import tree_pred, tree_grow, tree_grow_b, tree_pred_b, create_random_forest


# DATA IMPORT
training_data = pd.read_csv('eclipse-metrics-packages-2.0.csv', sep = ';')
testing_data = pd.read_csv('eclipse-metrics-packages-3.0.csv', sep = ';')

# PREPROCESSING
feat_41 = ['pre','ACD_avg','ACD_max','ACD_sum','FOUT_avg', 'FOUT_max','FOUT_sum', 'MLOC_avg','MLOC_max', 'MLOC_sum',
 'NBD_avg','NBD_max','NBD_sum','NOCU','NOF_avg','NOF_max','NOF_sum','NOI_avg','NOI_max','NOI_sum','NOM_avg',
 'NOM_max','NOM_sum','NOT_avg','NOT_max','NOT_sum','NSF_avg','NSF_max','NSF_sum','NSM_avg','NSM_max','NSM_sum','PAR_avg',
 'PAR_max','PAR_sum','TLOC_avg','TLOC_max','TLOC_sum','VG_avg','VG_max','VG_sum']

training_subset = training_data[feat_41]
testing_subset = testing_data[feat_41]

# create prediction column
y_train = (training_data['post'] > 0).astype(int)
y_test = (training_data['post'] > 0).astype(int)


# PERFORMANCE MEASURES
...

# 1. SINGLE CLASSIFICATION TREE
tree = tree_grow(training_subset, y_train, 15, 5, 41)
new_preds = tree_pred(testing_subset, tree)


# 2. BAGGING
bagging = tree_grow_b(training_subset, y_train, 15, 5, 41, 100)
new_preds = tree_pred_b(testing_subset, bagging)


# 3. RANDOM FOREST
randomforest = create_random_forest(training_subset, y_train, 15, 6, 41, 100)
new_preds = tree_pred_b(testing_subset, randomforest)