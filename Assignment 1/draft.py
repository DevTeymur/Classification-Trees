from matplotlib.pyplot import plt


def display_tree(node, depth=0):
    # Convert NumPy data types to Python types
    if isinstance(node, dict):
        if 'leaf' in node:
            # If this is a leaf node, print the class prediction
            print("  " * depth + f"Leaf: Class {node['class']}")
        else:
            # Print the feature and threshold for the split
            feature = node['feature']
            threshold = int(node['threshold']) if isinstance(node['threshold'], np.generic) else node['threshold']
            print("  " * depth + f"Feature {feature}, Threshold {threshold}")
            
            # Recursively print left and right branches
            print("  " * depth + "Left:")
            display_tree(node['left'], depth + 1)
            print("  " * depth + "Right:")
            display_tree(node['right'], depth + 1)
    elif isinstance(node, list):
        # If the node is a list (in case multiple trees), apply display_tree to each element
        for sub_tree in node:
            display_tree(sub_tree, depth)


def plot_decision_tree(tree, feature_names=None):
    fig, ax = plt.subplots(figsize=(10, 8))

    # Recursive helper function to draw tree
    def plot_node(node, x=0.5, y=1.0, dx=0.2, dy=0.1, depth=0):
        if 'leaf' in node:
            # Leaf node
            class_label = f"Class {node['class']}"
            ax.text(x, y, class_label, ha='center', va='center', fontsize=12, bbox=dict(facecolor='lightblue', edgecolor='black', boxstyle='round,pad=0.5'))
        else:
            # Decision node
            feature = node['feature']
            threshold = node['threshold']
            if feature_names:
                label = f"{feature_names[feature]} < {threshold}"
            else:
                label = f"Feature {feature} < {threshold}"
            
            ax.text(x, y, label, ha='center', va='center', fontsize=12, bbox=dict(facecolor='lightgreen', edgecolor='black', boxstyle='round,pad=0.5'))
            
            # Recursively plot left and right branches
            plot_node(node['left'], x - dx, y - dy, dx * 0.5, dy, depth + 1)
            plot_node(node['right'], x + dx, y - dy, dx * 0.5, dy, depth + 1)
            
            # Draw lines to children
            ax.plot([x, x - dx], [y - dy / 2, y - dy], 'k-')
            ax.plot([x, x + dx], [y - dy / 2, y - dy], 'k-')

    # Start plotting from the root of the tree
    plot_node(tree)

    # Set limits and hide axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.show()

