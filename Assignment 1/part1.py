import pandas as pd
import numpy as np
import random
import scipy

# Overall steps
# 1. Check if len is less than nmin -> return leaf node with majority class
# 2. Randomly select nfeat features then calculate the gini index (threshold)
# 3. Check the len of the samples in each lead node (right and left) to see that it meets the minleaf requirement, select the lowest gini index
# 4. Split the data into two groups based on the threshold, call tree_grow recursively on the two groups
# 5. Create a dictonary: selected feature, threshold, links to the left and right nodes
# 6. Return the dictonary for example: {'feature': 1, 'threshold': 0.5, 'left': left_node, 'right': right_node}


def tree_grow(x, y, nmin, minleaf, nfeat):
    n_samples = len(y)
    majority_class = int(np.bincount(y).argmax())

    # In both cases, we return a leaf node with the majority class
    if n_samples < nmin:
        return {'leaf': True, 'class': majority_class} # Get the majority class and return it
    if n_samples < minleaf:
        return {'leaf': True, 'class': majority_class} # Same for here, making a leaf node
    
    best_feat, best_threshold = find_best_split(x, y, nfeat)
    if best_feat is None:
        return {'leaf': True, 'class': majority_class} # No valid split found, return the majority class as a leaf node
    
    # Splitting the data into two groups based on the found best threshold
    left_indices, right_indices = x[:, best_feat] <= best_threshold, x[:, best_feat] > best_threshold

    # Recursively call tree_grow on the two groups to create the child nodes
    left_child = tree_grow(x[left_indices], y[left_indices], nmin, minleaf, nfeat)
    right_child = tree_grow(x[right_indices], y[right_indices], nmin, minleaf, nfeat)

    return {
        'feature': best_feat,
        'threshold': best_threshold,
        'left': left_child,
        'right': right_child,
    }
    

def tree_pred(x, tr):
    pred_results = []

    for sample in x:
        curr_node = tr

        # Check if the current node is a leaf and iterate until we reach a leaf node
        while 'class' not in curr_node:
            feature, threshold = curr_node['feature'], curr_node['threshold']
            # Check with threshold and move to the corresponding child node
            if sample[feature] <= threshold:
                curr_node = curr_node['left']
            else:
                curr_node = curr_node['right']
        # We have reached a leaf node, append the class to the results
        pred_results.append(curr_node['class'])

    return np.array(pred_results)
        

def tree_grow_b(x, y, nmin, minleaf, nfeat, m):
    tree_obj_list = []

    for _ in range(m):
        # Generating random indices for the bootstrap sample
        bootstrap_indices = np.random.choice(range(len(y)), size=len(y), replace=True)
        # Taking this random values to fit to the tree
        x_bootstrap, y_bootstrap = x[bootstrap_indices], y[bootstrap_indices]

        tree_obj = tree_grow(x_bootstrap, y_bootstrap, nmin, minleaf, nfeat)
        tree_obj_list.append(tree_obj)

    return tree_obj_list


def tree_pred_b(x, tree_obj_list):
    pred_results = []

    for tree_obj in tree_obj_list:
        # Predict and append to the pred results list which is 2D array
        pred_results.append(tree_pred(x, tree_obj))

    #
    pred_results = np.array(pred_results).T

    # In order to take majority votes we use mode
    final_preds = scipy.stats.mode(pred_results, axis=1)[0].flatten()
    return final_preds


def calc_gini_index(y, logs=False):
    len_target = len(y)
    classes, counts = np.unique(y, return_counts=True)
    gini_index = 1 - sum([(count/len_target)**2 for count in counts])
    print(f'Classes: {classes}, Counts: {counts}, Gini Index: {gini_index}') if logs else None
    return gini_index


def find_best_split(x, y, nfeat, logs=False):
    curr_num_features = x.shape[1]
    # Random selection of nfeat features
    selected_features = random.sample(range(curr_num_features), nfeat)
    print(f'Selected Features: {selected_features}') if logs else None
    # Calculate the gini index for each feature
    best_gini, best_feature, best_threshold = float('inf'), None, None

    for feature in selected_features:
        thresholds = np.unique(x[:, feature]) # All distinct values of the current feature
        for threshold in thresholds:
            left_indices, right_indices = x[:, feature] <= threshold, x[:, feature] > threshold
            
            # Check if there is enough samples in the left and right nodes
            if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                continue

            # Calc of the gini indexes for left and right nodes
            left_gini, right_gini = calc_gini_index(y[left_indices]), calc_gini_index(y[right_indices])
            weighted_gini = (np.sum(left_indices) * left_gini + np.sum(right_indices) * right_gini) / len(y)

            if weighted_gini < best_gini:
                best_gini, best_feature, best_threshold = weighted_gini, feature, threshold

    print(f'Best Gini: {best_gini}, Best Feature: {best_feature}, Best Threshold: {best_threshold}') if logs else None
    return best_feature, best_threshold


def calc_metrics():
    # Accuracy, Precision, Recall
    pass

def create_confusion_matrix():
    pass


# Call of the functions
credit_data = pd.read_csv('credit_data.csv')
x = credit_data.drop('class', axis=1)
y = credit_data['class']

# print(x, y)

result_tree = tree_grow(x.values, y.values, 2, 1, 4)
print(result_tree)

new_preds = tree_pred(x.values, result_tree)
print(new_preds)

result_trees = tree_grow_b(x.values, y.values, 2, 1, 4, 5)
print(result_trees)

new_preds_b = tree_pred_b(x.values, result_trees)
print(new_preds_b)