import numpy as np
from sklearn.utils import shuffle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, accuracy_score, recall_score, confusion_matrix


class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def gini_index(y):
    _, counts = np.unique(y, return_counts=True)
    proportions = counts / len(y)
    return 1 - np.sum(proportions**2)

def best_split(x, y, minleaf, nfeat):
    num_features = x.shape[1]
    best_gini = float('inf')
    best_feature = None
    best_threshold = None
    best_left_y = None
    best_right_y = None

    features = np.random.choice(num_features, nfeat, replace=False)
    
    for feature in features:
        thresholds = np.unique(x[:, feature])
        for threshold in thresholds:
            left_mask = x[:, feature] <= threshold
            right_mask = x[:, feature] > threshold
            
            if np.sum(left_mask) < minleaf or np.sum(right_mask) < minleaf:
                continue
            
            left_y, right_y = y[left_mask], y[right_mask]
            gini = (len(left_y) * gini_index(left_y) + len(right_y) * gini_index(right_y)) / len(y)
            
            if gini < best_gini:
                best_gini = gini
                best_feature = feature
                best_threshold = threshold
                best_left_y = left_y
                best_right_y = right_y

    return best_feature, best_threshold, best_left_y, best_right_y

def tree_grow(x, y, nmin, minleaf, nfeat):
    if len(y) < nmin:
        return TreeNode(value=np.mean(y).round().astype(int))
    
    feature, threshold, left_y, right_y = best_split(x, y, minleaf, nfeat)
    
    if feature is None:
        return TreeNode(value=np.mean(y).round().astype(int))
    
    left_node = tree_grow(x[x[:, feature] <= threshold], left_y, nmin, minleaf, nfeat)
    right_node = tree_grow(x[x[:, feature] > threshold], right_y, nmin, minleaf, nfeat)
    
    return TreeNode(feature, threshold, left_node, right_node)

def tree_pred(x, tree):
    def predict_node(row, node):
        if node.value is not None:
            return node.value
        if row[node.feature] <= node.threshold:
            return predict_node(row, node.left)
        else:
            return predict_node(row, node.right)
    
    predictions = [predict_node(row, tree) for row in x]
    return np.array(predictions)

def tree_grow_b(x, y, nmin, minleaf, nfeat, m):
    trees = []
    for _ in range(m):
        x_bootstrap, y_bootstrap = shuffle(x, y, random_state=None)
        x_bootstrap, y_bootstrap = x_bootstrap[:len(x)], y_bootstrap[:len(y)]
        tree = tree_grow(x_bootstrap, y_bootstrap, nmin, minleaf, nfeat)
        trees.append(tree)
    return trees

def tree_pred_b(x, trees):
    def majority_vote(predictions):
        return np.bincount(predictions).argmax()
    
    all_predictions = np.array([tree_pred(x, tree) for tree in trees])
    majority_predictions = np.apply_along_axis(majority_vote, 0, all_predictions)
    
    return majority_predictions



def calc_metrics(y_true, y_pred):
    accuracy = round(accuracy_score(y_true, y_pred), 2)
    precision = round(precision_score(y_true, y_pred, average='binary'), 2)
    recall = round(recall_score(y_true, y_pred, average='binary'), 2)
    
    print(f'Accuracy = {accuracy}, Precision = {precision}, Recall = {recall}')    
    return accuracy, precision, recall


def create_confusion_matrix(y_true, y_pred, display=False):
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    print("Confusion Matrix:")
    print(f"{'':<12}{'Predicted 0':<12}{'Predicted 1':<12}")
    print(f"{'Actual 0':<12}{conf_matrix[0, 0]:<12}{conf_matrix[0, 1]:<12}")
    print(f"{'Actual 1':<12}{conf_matrix[1, 0]:<12}{conf_matrix[1, 1]:<12}")
    
    if display:
        plt.figure(figsize=(6,4))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, 
                    xticklabels=['Predicted 0', 'Predicted 1'], 
                    yticklabels=['Actual 0', 'Actual 1'])
        
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix Heatmap')
        plt.show()



# print('Credit data tree and forest construction')
# credit_data = pd.read_csv('credit_data.csv')
# x = credit_data.drop('class', axis=1).values
# y = credit_data['class'].values

# print('Creating a single tree:')
# result_tree = tree_grow(x, y, 2, 1, 5)
# new_preds = tree_pred(x, result_tree)
# # print(result_tree)
# # print(new_preds)

# calc_metrics(y, new_preds)
# create_confusion_matrix(y, new_preds)

# print('\nCreating a forest of 5 trees:')
# result_trees = tree_grow_b(x, y, 2, 1, 5, 5)
# new_preds_b = tree_pred_b(x, result_trees)

# print(result_trees)
# print(new_preds_b)
# # calc_metrics(y, new_preds_b)
# # create_confusion_matrix(y, new_preds_b)


print("__" * 50)
print('\nPima Indians dataset tree construction')
pima_data = pd.read_csv('pima_indians_data.csv', header=None)
columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'class']
pima_data.columns = columns

# print(pima_data.head())
x = pima_data.drop('class', axis=1).values
y = pima_data['class'].values

# print(x, y)
print('Creating a single tree:')
result_tree = tree_grow(x, y, 20, 5, 8)
new_preds = tree_pred(x, result_tree)

# # print(result_tree)
# # print(new_preds)
calc_metrics(y, new_preds)
create_confusion_matrix(y, new_preds)

print('\nCreating a forest of 5 trees:')
result_trees = tree_grow_b(x, y, 20, 5, 8, 5)
new_preds_b = tree_pred_b(x, result_trees)
# print(result_trees)
# print(new_preds_b)

calc_metrics(y, new_preds_b)
create_confusion_matrix(y, new_preds_b)

# print('\nCreating a Random Forest of 5 trees:')
# result_rf = create_random_forest(x, y, n_trees, 20, 5, 8)
# new_preds_rf = tree_pred_b(x, result_rf)
# calc_metrics(y, new_preds_rf)
# create_confusion_matrix(y, new_preds_rf)