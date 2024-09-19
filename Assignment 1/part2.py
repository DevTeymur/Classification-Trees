import pandas as pd
import numpy as np
from final import tree_grow, tree_grow_b, tree_pred, tree_pred_b, calc_metrics, create_confusion_matrix

import warnings
warnings.filterwarnings("ignore")

eclipse_data_2 = pd.read_csv("eclipse-metrics-packages-2.0.csv",sep=';')
eclipse_data_3 = pd.read_csv("eclipse-metrics-packages-3.0.csv",sep=';')

selected_features = [
 'post',
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
 'VG_sum']

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

print('_____'*10)
print('Bagging results:')
result_trees = tree_grow_b(X_train, y_train, nmin=15, minleaf=5, nfeat=41, m=100)
y_pred_b = tree_pred_b(x_test, result_trees)

acc, prec, recall = calc_metrics(y_test, y_pred_b)
create_confusion_matrix(y_test, y_pred_b, display=True, title='Bagging Confusion Matrix')

print('_____'*10)
print('Creating random forest:')
result_trees = tree_grow_b(X_train, y_train, nmin=15, minleaf=5, nfeat=6, m=round(np.sqrt(41)))
y_pred_b = tree_pred_b(x_test, result_trees)

acc, prec, recall = calc_metrics(y_test, y_pred_b)
create_confusion_matrix(y_test, y_pred_b, display=True, title='Random Forest Confusion Matrix')

