# Amber Koelfat - 6467296
# Max Verweij - 6791409
# Orane Pereira - 7644701
# Teymur Rzali - 4625471

import scipy
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, accuracy_score, recall_score, confusion_matrix

np.random.seed(0)

def tree_grow(x, y, nmin, minleaf, nfeat):
    """
    Recursively growing a decision tree based on the given data.

    Overall steps briefly:
    1. Check if len is less than nmin -> return leaf node with majority class
    2. Randomly select nfeat features then calculate the gini index (threshold)
    3. Check the len of the samples in each lead node (right and left) to see that it meets the minleaf requirement, select the lowest gini index
    4. Split the data into two groups based on the threshold, call tree_grow recursively on the two groups
    5. Create a dictonary: selected feature, threshold, links to the left and right nodes
    6. Return the dictonary for example: {'feature': 1, 'threshold': 0.5, 'left': left_node, 'right': right_node}

    Args:
        x (numpy.ndarray): Feature matrix (rows of the data).
        y (numpy.ndarray): Target vector.
        nmin (int): Minimum number of samples required to split a node.
        minleaf (int): Minimum number of samples required in a leaf node.
        nfeat (int): Number of random features to consider for each split.

    Returns:
        dict: A dictoinary which represents the tree sturcture. Contains feature, its threshold, 
        left and right nodes which also contains the same structure or a leaf node. 
    """
    n_samples = len(y)
    majority_class = int(np.bincount(y).argmax()) # Finding the most common class in y
    class_distribution = dict(zip(*np.unique(y, return_counts=True)))

    # In both cases, we return a leaf node with the majority class
    if n_samples < nmin:# or len(np.unique(y)) == 1:
        return {
            'leaf': True,
            'class': majority_class,
            'samples': n_samples,
            'class_distribution': class_distribution
        } # Get the majority class and return it
    
    best_feat, best_threshold = find_best_split(x, y, minleaf, nfeat)
    if best_feat is None:
        return {
            'leaf': True,
            'class': majority_class,
            'samples': n_samples,
            'class_distribution': class_distribution
        } # No valid split found, return the majority class as a leaf node
    
    # Splitting the data into two groups based on the found best threshold
    left_indices, right_indices = x[:, best_feat] <= best_threshold, x[:, best_feat] > best_threshold

    # Check if either side of the split is empty
    if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
        return {
            'leaf': True,
            'class': majority_class,
            'samples': n_samples,
            'class_distribution': class_distribution
        } # If empty, stop growing and return majority class
    
    # Recursively call tree_grow on the two groups to create the child nodes
    left_child = tree_grow(x[left_indices], y[left_indices], nmin, minleaf, nfeat)
    right_child = tree_grow(x[right_indices], y[right_indices], nmin, minleaf, nfeat)

    return {
        'feature': best_feat,
        'threshold': best_threshold,
        'samples': n_samples,
        'class_distribution': class_distribution,
        'left': left_child,
        'right': right_child
    }
    

def tree_pred(x, tr):
    """
    A predictor function which takes the tree that been created for a set of samples.

    Args:
        x (numpy.ndarray): Feature matrix (rows of the data).
        tr (dict): output dictionary from the tree_grow function

    Returns:
        numpy.ndarray: An array of the predicted results for the given samples.
    """
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
        # Iteration ended, we have reached a leaf node, append the class to the results
        pred_results.append(curr_node['class'])

    return np.array(pred_results)
        

def tree_grow_b(x, y, nmin, minleaf, nfeat, m):
    """
    Builds a forest of the trees using the bagging technique.

    Args:
        x (numpy.ndarray): Feature matrix (rows of the data).
        y (numpy.ndarray): Target vector.
        nmin (int): Minimum number of samples to allow a split.
        minleaf (int): Minimum number of samples required to create a leaf node.
        nfeat (int): Number of features to consider when looking for the best split.
        m (int): Number of trees to grow in the forest.

    Returns:
        list: list of the decision trees that have been created with tree_grow function.
    """
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
    """
    Predicts class labels using a forest of decision trees (bagging).

    Args:
        x (numpy.ndarray): Feature matrix (rows of the data).
        tree_obj_list (list): List of trained decision trees.

    Returns:
        numpy.ndarray: Array of class labels based on majority voting across trees.
    """
    pred_results = []

    for tree_obj in tree_obj_list:
        # Predict and append to the pred results list which is 2D array
        pred_results.append(tree_pred(x, tree_obj))

    # Transpose to organize preds such that each row is a single sample's predictions across all trees
    pred_results = np.array(pred_results).T

    # In order to take majority votes we use mode
    final_preds = scipy.stats.mode(pred_results, axis=1)[0].flatten()
    return final_preds


def find_best_split(x, y, minleaf, nfeat, logs=False):
    """
    Finds the best split for the given dataset based on the Gini index.

    Args:
        x (numpy.ndarray): Feature matrix (rows of data).
        y (numpy.ndarray): Target labels.
        nfeat (int): Number of features to randomly select for the split.
        logs (bool, optional): If True, prints the selected features and details about the best split. Defaults to False.

    Returns:
        tuple: The best feature index arnd best threshold value for the split.
               If no valid split is found, returns (None, None).
    """
    curr_num_features = x.shape[1]
    # Random selection of nfeat features to consider for the split
    # selected_features = random.sample(range(curr_num_features), nfeat)
    selected_features = np.random.choice(curr_num_features, nfeat, replace=False)
    print(f'Selected Features: {selected_features}') if logs else None

    best_gini, best_feature, best_threshold = float('inf'), None, None

    for feature in selected_features:
        # From the current feature, get the unique values to consider as thresholds
        thresholds = np.sort(np.unique(x[:, feature]))  # Sort thresholds to ensure consistency

        for threshold in thresholds:
            left_indices, right_indices = x[:, feature] <= threshold, x[:, feature] > threshold # Splitting
            
            # # Check if either side of the split is empty
            if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                continue

            if np.sum(left_indices) < minleaf or np.sum(right_indices) < minleaf:
                continue
            
            # Left, right gini calculation and based on the calculated values weighted gini is calculated
            left_gini, right_gini = calc_gini_index(y[left_indices]), calc_gini_index(y[right_indices])
            weighted_gini = (np.sum(left_indices) * left_gini + np.sum(right_indices) * right_gini) / len(y)

            # Update the best feature and threshold if we get lower gini index
            if weighted_gini < best_gini:
                best_gini, best_feature, best_threshold = weighted_gini, feature, threshold

    # If no valid split found, return None, None
    if best_feature is None or best_threshold is None:
        return None, None

    print(f'Best Gini: {round(best_gini, 2)}, Best Feature: {best_feature}, Best Threshold: {best_threshold}') if logs else None
    return best_feature, int(best_threshold)


def calc_gini_index(y, logs=False):
    """
    A function to calculate the Gini index for a given set of class labels.

    Args:
        y (numpy.ndarray): Array of class labels.
        logs (bool, optional): If True, prints details about the Gini index calculation. Defaults to False.

    Returns:
        float: Gini index, representing the impurity of the set.
    """
    len_target = len(y)
    classes, counts = np.unique(y, return_counts=True)
    gini_index = 1 - sum([(count/len_target)**2 for count in counts])
    print(f'Classes: {classes}, Counts: {counts}, Gini Index: {gini_index}') if logs else None
    return gini_index


def calc_metrics(y_true, y_pred):
    """
    Calculates and prints the accuracy, precision, and recall of the model predictions.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.

    Returns:
        tuple: A tuple containing accuracy, precision, and recall, each rounded to 2 decimal places.
    """
    accuracy = round(accuracy_score(y_true, y_pred), 2)
    precision = round(precision_score(y_true, y_pred, average='binary'), 2)
    recall = round(recall_score(y_true, y_pred, average='binary'), 2)
    
    print(f'Accuracy = {accuracy}, Precision = {precision}, Recall = {recall}')    
    return accuracy, precision, recall


def create_confusion_matrix(y_true, y_pred, display=False, title='Confusion Matrix Heatmap'):
    """
    Creates and prints the confusion matrix for the given true and predicted labels. 
    Optionally displays a heatmap visualization of the confusion matrix using seaborn package.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        display (bool): If True, displays the confusion matrix as a heatmap. Defaults to False.
        title (str): Title of the heatmap. Defaults to 'Confusion Matrix Heatmap'.
        
    Returns:
        None: This function does not return any value, but it prints the confusion matrix and optionally shows a heatmap.
    """
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
        plt.title(title)
        plt.savefig(f'img/{title}.png')
        # plt.show()


def print_splits(tree, depth=0, max_depth=3):
    """
    Recursively prints the splits of a decision tree in a readable format.

    Args:
        tree (dict): The decision tree to traverse.
        depth (int): The current depth (default is 0).
        max_depth (int): The maximum depth to print (default is 3).

    Returns:
        None: This function does not return any value, but it prints the splits of the decision tree.
    """
    # If we've reached a leaf node or exceeded the max depth, stop
    if 'leaf' in tree or depth >= max_depth:
        return

    # Print the current split (feature and threshold) with indentation based on depth
    indent = "  " * depth  # Indentation for readability
    print(f"{indent}Depth {depth}: Split on feature '{tree['feature']}' at threshold {tree['threshold']}. "
          f"Samples: {tree['samples']}, Class Distribution: {tree['class_distribution']}")
    
    # Recursively print for the left and right child nodes
    print_splits(tree['left'], depth + 1, max_depth)
    print_splits(tree['right'], depth + 1, max_depth)


def get_first_three_levels(tree, selected_features):
    """
    Extracts the first three levels from the decision tree including root, its left and right children,
    and their respective children.

    Args:
        tree (dict): The decision tree structure.
        selected_features (list): List of feature names corresponding to feature indices.

    Returns:
        dict: A dictionary with the first three levels (root, left child, right child, their children) 
              including feature, threshold, samples, and class distribution.
    """
    if tree is None:
        return None

    # Extract root level
    root_split = {
        "feature": selected_features[tree.get("feature")],
        "threshold": tree.get("threshold"),
        "samples": tree.get("samples"),
        "class_distribution": tree.get("class_distribution"),
        "left": None,
        "right": None
    }

    # Extract left child
    left_child = tree.get("left")
    if left_child is not None:
        root_split["left"] = {
            "feature": selected_features[left_child.get("feature")],
            "threshold": left_child.get("threshold"),
            "samples": left_child.get("samples"),
            "class_distribution": left_child.get("class_distribution"),
            "left": None,
            "right": None
        }

        # Extract left child's children
        left_left_child = left_child.get("left")
        left_right_child = left_child.get("right")
        if left_left_child is not None:
            root_split["left"]["left"] = {
                "feature": selected_features[left_left_child.get("feature")],
                "threshold": left_left_child.get("threshold"),
                "samples": left_left_child.get("samples"),
                "class_distribution": left_left_child.get("class_distribution"),
            }
        if left_right_child is not None:
            root_split["left"]["right"] = {
                "feature": selected_features[left_right_child.get("feature")],
                "threshold": left_right_child.get("threshold"),
                "samples": left_right_child.get("samples"),
                "class_distribution": left_right_child.get("class_distribution"),
            }

    # Extract right child
    right_child = tree.get("right")
    if right_child is not None:
        root_split["right"] = {
            "feature": selected_features[right_child.get("feature")],
            "threshold": right_child.get("threshold"),
            "samples": right_child.get("samples"),
            "class_distribution": right_child.get("class_distribution"),
            "left": None,
            "right": None
        }

        # Extract right child's children
        right_left_child = right_child.get("left")
        right_right_child = right_child.get("right")
        if right_left_child is not None:
            root_split["right"]["left"] = {
                "feature": selected_features[right_left_child.get("feature")],
                "threshold": right_left_child.get("threshold"),
                "samples": right_left_child.get("samples"),
                "class_distribution": right_left_child.get("class_distribution"),
            }
        if right_right_child is not None:
            root_split["right"]["right"] = {
                "feature": selected_features[right_right_child.get("feature")],
                "threshold": right_right_child.get("threshold"),
                "samples": right_right_child.get("samples"),
                "class_distribution": right_right_child.get("class_distribution"),
            }

    # Return the structure for the first three levels
    return root_split

# Call of the functions
if __name__ == '__main__':
    print("__" * 50)
    print('Credit data tree and forest construction')
    credit_data = pd.read_csv('credit_data.csv')
    x = credit_data.drop('class', axis=1).values
    y = credit_data['class'].values
    nmin, minleaf, nfeat, m = 2, 1, 5, 5
    n_trees = 5

    print('Creating a single tree:')
    result_tree = tree_grow(x, y, nmin, minleaf, nfeat)
    new_preds = tree_pred(x, result_tree)

    # print(result_tree)
    # print(new_preds)
    calc_metrics(y, new_preds)
    create_confusion_matrix(y, new_preds)

    print('\nCreating a forest of 5 trees:')
    result_trees = tree_grow_b(x, y, nmin, minleaf, nfeat, m)
    new_preds_b = tree_pred_b(x, result_trees)

    # print(result_trees)
    # print(new_preds_b)
    calc_metrics(y, new_preds_b)
    create_confusion_matrix(y, new_preds_b)
    # print_first_splits(result_tree)

    # _____________________________________________________________________________________________________
    # Testing on pima indians dataset
    print("__" * 50)
    print('\nPima Indians dataset tree construction')
    pima_data = pd.read_csv('pima_indians_data.csv', header=None)
    columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'class']
    pima_data.columns = columns
    nmin, minleaf, nfeat, m = 20, 5, 8, 5
    n_trees = 5

    x = pima_data.drop('class', axis=1).values
    y = pima_data['class'].values

    print('Creating a single tree:')
    result_tree = tree_grow(x, y, nmin, minleaf, nfeat)
    new_preds = tree_pred(x, result_tree)

    # print(result_tree)
    # print(new_preds)
    calc_metrics(y, new_preds)
    create_confusion_matrix(y, new_preds)

    print('\nCreating a forest of 5 trees:')
    result_trees = tree_grow_b(x, y, nmin, minleaf, nfeat, m)
    new_preds_b = tree_pred_b(x, result_trees)

    # print(result_trees)
    # print(new_preds_b)
    calc_metrics(y, new_preds_b)
    create_confusion_matrix(y, new_preds_b)
