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

def repeated_tests(test_amount = 30):
  # single tree
  data = {'acc':[],
          'prec':[],
          'recall':[],
          'tree_type':[]}
  
  # print("decision tree")
  # decision tree
  for i in range(test_amount):
    print(X_train)
    result_tree = tree_grow(X_train, y_train, nmin=15, minleaf=5, nfeat=41)
    y_pred = tree_pred(x_test, result_tree)
  
    acc, prec, recall = calc_metrics(y_test, y_pred)

    add_to_data(data,acc,prec,recall,"single_tree")
  
  print("\nbagging")
  # bagging
  for i in range(test_amount):
    result_trees_b = tree_grow_b(X_train, y_train, nmin=15, minleaf=5, nfeat=41, m=100)
    y_pred_b = tree_pred_b(x_test, result_trees_b)
  
    acc, prec, recall = calc_metrics(y_test, y_pred_b)

    add_to_data(data,acc,prec,recall,"bagging_tree")
  
  # random forest
  print("\nrandom forest")
  for i in range(test_amount):
    result_trees_rf = tree_grow_b(X_train, y_train, nmin=15, minleaf=5, nfeat=6, m=round(np.sqrt(41)))
    y_pred_rf = tree_pred_b(x_test, result_trees_rf)
  
    acc, prec, recall = calc_metrics(y_test, y_pred_rf)

    add_to_data(data,acc,prec,recall,"random_forest")

  return pd.DataFrame(data=data,columns=['acc','prec','recall','tree_type'])
  

def add_to_data(data,acc,prec,recall,type):
  data['acc'].append(acc)
  data['prec'].append(prec)
  data['recall'].append(recall)
  data['tree_type'].append(type)

  
df = repeated_tests(30)
# df.to_csv("distribution_data.csv")
df = pd.read_csv("distribution_data.csv")

def plot_dist(metric='acc',bin_width=0.0025):
  sns.histplot(df[df['tree_type']=='single_tree'][metric], color='blue', label='Single Tree', kde=True,binwidth=bin_width)
  sns.histplot(df[df['tree_type']=='bagging_tree'][metric], color='green', label='Bagging', kde=True,binwidth=bin_width)
  sns.histplot(df[df['tree_type']=='random_forest'][metric], color='red', label='Random Forest', kde=True,binwidth=bin_width)
  
  # Add plot title and legend
  plt.title(f'Distribution of {metric}')
  plt.legend()
  
  # Show the plot
  plt.savefig(f'img/distributions/{metric} distributions.png')
  plt.clf()

def plot_boxplot(metric='acc'):
  sns.boxplot(x='tree_type', y=metric, data=df, color='blue')
  
  plt.savefig(f'img/distributions/{metric} boxplot.png')
  plt.clf()

"""
for metric in ['acc','prec','recall']:
  plot_dist(metric)
  plot_boxplot(metric)

df_st = df[df['tree_type'] == 'single_tree']
df_bt = df[df['tree_type'] == 'bagging_tree']
df_rf = df[df['tree_type'] == 'random_forest']

print(ttest_rel(df_st['acc'],df_rf['acc'],alternative='two-sided'))
print(ttest_rel(df_bt['acc'],df_st['acc'],alternative='two-sided'))
print(ttest_rel(df_bt['acc'],df_rf['acc'],alternative='two-sided'))

print(wilcoxon(df_st['acc'],df_rf['acc']))
print(wilcoxon(df_bt['acc'],df_st['acc']))
print(wilcoxon(df_rf['acc'],df_bt['acc']))

print(f_oneway(df_rf['acc'],df_st['acc'],df_bt['acc']))
"""