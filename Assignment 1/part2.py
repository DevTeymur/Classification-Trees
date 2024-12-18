import pandas as pd
import numpy as np
from final import *

from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings("ignore")

eclipse_data_2 = pd.read_csv("eclipse-metrics-packages-2.0.csv",sep=';')
eclipse_data_3 = pd.read_csv("eclipse-metrics-packages-3.0.csv",sep=';')

selected_features = [
 'pre',
 'ACD_avg',
 'ACD_max',
 'ACD_sum',
 'FOUT_avg',
 'FOUT_max',
 'FOUT_sum',
 'MLOC_avg',
 'MLOC_max',
 'MLOC_sum',
 'NBD_avg',
 'NBD_max',
 'NBD_sum',
 'NOCU',
 'NOF_avg',
 'NOF_max',
 'NOF_sum',
 'NOI_avg',
 'NOI_max',
 'NOI_sum',
 'NOM_avg',
 'NOM_max',
 'NOM_sum',
 'NOT_avg',
 'NOT_max',
 'NOT_sum',
 'NSF_avg',
 'NSF_max',
 'NSF_sum',
 'NSM_avg',
 'NSM_max',
 'NSM_sum',
 'PAR_avg',
 'PAR_max',
 'PAR_sum',
 'TLOC_avg',
 'TLOC_max',
 'TLOC_sum',
 'VG_avg',
 'VG_max',
 'VG_sum',
 'post',
 ]

# Selecting the relevant features
training_data = eclipse_data_2[selected_features]
test_data = eclipse_data_3[selected_features]

# Creating a class label that is 1 when 'post' > 0 and 0 otherwise
training_data.loc[:, 'class_label'] = (training_data['post'] > 0).astype(int)
X_train = training_data.drop(['class_label','post'], axis=1).values
y_train = training_data['class_label'].values

# Creating a class label that is 1 when 'post' > 0 and 0 otherwise
test_data['class_label'] = (test_data['post'] > 0).astype(int)
x_test = test_data.drop(['class_label','post'], axis=1).values
y_test = test_data['class_label'].values

print('Creating a single tree:')
result_tree = tree_grow(X_train, y_train, nmin=15, minleaf=5, nfeat=41)
y_pred = tree_pred(x_test, result_tree)

acc, prec, recall = calc_metrics(y_test, y_pred)
create_confusion_matrix(y_test, y_pred, display=True, title='Single Tree Confusion Matrix')

# Example usage
first_three_splits = get_first_three_levels(result_tree, selected_features)
# print(first_three_splits)
# print('_____'*10)
# print(result_tree)


print('_____'*10)
print('Bagging results:')
result_trees_b = tree_grow_b(X_train, y_train, nmin=15, minleaf=5, nfeat=41, m=100)
y_pred_b = tree_pred_b(x_test, result_trees_b)

acc, prec, recall = calc_metrics(y_test, y_pred_b)
create_confusion_matrix(y_test, y_pred_b, display=True, title='Bagging Confusion Matrix')

print('_____'*10)
print('Creating random forest:')
result_trees_rf = tree_grow_b(X_train, y_train, nmin=15, minleaf=5, nfeat=6, m=round(np.sqrt(41)))
y_pred_rf = tree_pred_b(x_test, result_trees_rf)

acc, prec, recall = calc_metrics(y_test, y_pred_rf)
create_confusion_matrix(y_test, y_pred_rf, display=True, title='Random Forest Confusion Matrix')


# _____________________________________________________________________________
# McNemar's Test - not completed

# from sklearn.metrics import confusion_matrix
# from statsmodels.stats.contingency_tables import mcnemar

# contingency_tree_bagging = confusion_matrix(y_pred, y_pred_b)
# contingency_tree_rf = confusion_matrix(y_pred, y_pred_rf)
# contingency_bagging_rf = confusion_matrix(y_pred_b, y_pred_rf)

# # McNemar Test
# result_tree_bagging = mcnemar(contingency_tree_bagging, exact=True)
# result_tree_rf = mcnemar(contingency_tree_rf, exact=True)
# result_bagging_rf = mcnemar(contingency_bagging_rf, exact=True)

# # Print p-values
# print(f'McNemar Test p-value (Tree vs. Bagging): {result_tree_bagging.pvalue}')
# print(f'McNemar Test p-value (Tree vs. Random Forest): {result_tree_rf.pvalue}')
# print(f'McNemar Test p-value (Bagging vs. Random Forest): {result_bagging_rf.pvalue}')

from statsmodels.stats.contingency_tables import mcnemar

# Function to build the contingency table for McNemar's test
def build_mcnemar_table(y_true, y_pred1, y_pred2):
    b = np.sum((y_pred1 == y_true) & (y_pred2 != y_true))  # Model 1 correct, Model 2 wrong
    c = np.sum((y_pred1 != y_true) & (y_pred2 == y_true))  # Model 1 wrong, Model 2 correct
    return [[0, b], [c, 0]]

# McNemar's Test between Tree and Bagging
contingency_tree_bagging = build_mcnemar_table(y_test, y_pred, y_pred_b)
result_tree_bagging = mcnemar(contingency_tree_bagging, exact=True)
print(f'McNemar Test p-value (Tree vs. Bagging): {result_tree_bagging.pvalue}')

# McNemar's Test between Tree and Random Forest
contingency_tree_rf = build_mcnemar_table(y_test, y_pred, y_pred_rf)
result_tree_rf = mcnemar(contingency_tree_rf, exact=True)
print(f'McNemar Test p-value (Tree vs. Random Forest): {result_tree_rf.pvalue}')

# McNemar's Test between Bagging and Random Forest
contingency_bagging_rf = build_mcnemar_table(y_test, y_pred_b, y_pred_rf)
result_bagging_rf = mcnemar(contingency_bagging_rf, exact=True)
print(f'McNemar Test p-value (Bagging vs. Random Forest): {result_bagging_rf.pvalue}')
