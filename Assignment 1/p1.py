import numpy as np
from sklearn.utils import shuffle

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
