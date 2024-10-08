import pandas as pd
import numpy as np
from final import tree_grow, tree_grow_b, tree_pred, tree_pred_b, calc_metrics, create_confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel,wilcoxon,normaltest,f_oneway


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

def single_tests():
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

def add_to_data(data,acc,prec,recall,type):
  data['acc'].append(acc)
  data['prec'].append(prec)
  data['recall'].append(recall)
  data['tree_type'].append(type)

  
# # df = repeated_tests(30)
# df.to_csv("distribution_data.csv")
# df = pd.read_csv("distribution_data.csv")

def plot_dist(df, bin_width=0.0025):
  sns.histplot(df['st'], color='blue', label='Single Tree', kde=True,binwidth=bin_width)
  sns.histplot(df['bt'], color='green', label='Bagging', kde=True,binwidth=bin_width)
  sns.histplot(df['rf'], color='red', label='Random Forest', kde=True,binwidth=bin_width)
  
  # Add plot title and legend
  plt.title(f'Distribution of Acc')
  plt.legend()
  
  # Show the plot
  plt.savefig(f'img/distributions/Acc distributions.png')
  plt.clf()

def plot_boxplot(df):
  fig, axes = plt.subplots(1, 3, figsize=(18, 6))
  
  for column, ax in zip(df.columns,axes):
    sns.boxplot(data=df[column], color='blue', ax=ax)
  
  plt.savefig(f'img/distributions/acc boxplot.png')
  plt.clf()


def sample_data(test_df, n_test_sets=30):
  test_datasets = []
  test_set_size = len(test_df) // n_test_sets

  print(test_set_size)

  for i in range(n_test_sets):
    test_set = test_df.sample(n=test_set_size, replace=False)
    test_datasets.append(test_set)

  return test_datasets

  
def test_with_samples():
  sample_test_data = sample_data(test_data)

  acc_results_dict = {"st":[],
                "bt":[],
                "rf":[]}
  
  result_tree_st = tree_grow(X_train, y_train, nmin=15, minleaf=5, nfeat=41)
  result_trees_bt = tree_grow_b(X_train, y_train, nmin=15, minleaf=5, nfeat=41, m=100)
  result_trees_rf = tree_grow_b(X_train, y_train, nmin=15, minleaf=5, nfeat=6, m=round(np.sqrt(41)))
  
  for test_sample in sample_test_data:
    # Creating a class label that is 1 when 'post' > 0 and 0 otherwise
    test_sample['class_label'] = (test_sample['post'] > 0).astype(int)
    x_test_sample = test_sample.drop(['class_label','post'], axis=1).values
    y_test_sample = test_sample['class_label'].values
  
    # st
    y_pred = tree_pred(x_test_sample, result_tree_st)
    
    acc, prec, recall = calc_metrics(y_test_sample, y_pred)
    acc_results_dict['st'].append(acc)
  
    # bt
    y_pred_b = tree_pred_b(x_test_sample, result_trees_bt)
    
    acc, prec, recall = calc_metrics(y_test_sample, y_pred_b)
    acc_results_dict['bt'].append(acc)
    
    # rf
    y_pred_b = tree_pred_b(x_test_sample, result_trees_rf)
    
    acc, prec, recall = calc_metrics(y_test_sample, y_pred_b)
    acc_results_dict['rf'].append(acc)

    return pd.DataFrame(acc_results_dict)

results_df = test_with_samples()
results_df.to_csv("sampled_distribution_data.csv")

results_df = pd.read_csv("sampled_distribution_data.csv")

# plot_dist(results_df,0.039)
# plot_boxplot(results_df)
  