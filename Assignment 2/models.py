import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('data/cleaned_data.csv')
X, y= df['cleaned_text'], df['class']
X, y = shuffle(X, y, random_state=42)

# Split the data into training (folds 1-4) and testing (fold 5)
X_train = X[:640]  # First 640 reviews for training
y_train = y[:640]
X_test = X[640:]   # Last 160 reviews for testing
y_test = y[640:]


def remove_sparse_terms(X, threshold=0.05):
    # Calculate the occurrence of each feature across all samples
    feature_counts = X.sum(axis=0).A1  # Convert to 1D array
    total_samples = X.shape[0]
    
    # Keep features that occur in at least 5% of the samples
    mask = feature_counts >= (threshold * total_samples)
    return X[:, mask], mask


def create_matrix(X_train, X_test, y_train,y_test, type='unigram', remove_sparse=False):
    vectorizer = CountVectorizer(ngram_range=(1, 2) if type == 'bigram' else (1, 1))
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    
    if remove_sparse:
        print('Removing sparse terms...')
        X_train_filtered, mask = remove_sparse_terms(X_train_vectorized)
        X_test_filtered = X_test_vectorized[:, mask]  # Filter test data using the same mask
    else:
        X_train_filtered = X_train_vectorized
        X_test_filtered = X_test_vectorized

    return X_train_filtered, X_test_filtered, y_train, y_test, vectorizer


def naive_bayes(X_train, y_train, X_test, y_test, n_features=360, type='unigram', remove_sparse=True, logs=False):
    print('_____' * 20)
    print(f'Naive Bayes {type} Model')

    X_train_filtered, X_test_filtered, y_train, y_test, vectorizer = \
        create_matrix(X_train, X_test, y_train, y_test, type, remove_sparse)
    
    # Feature Selection using SelectKBest
    feature_selector = SelectKBest(chi2, k=n_features)
    X_train_selected = feature_selector.fit_transform(X_train_filtered, y_train)
    X_test_selected = feature_selector.transform(X_test_filtered)

    # Hyperparameter tuning using cross-validation
    nb_model = MultinomialNB()
    param_grid = {'alpha': np.arange(0.1, 5.1, 0.1)}

    cv = StratifiedKFold(n_splits=5)

    # Grid search for best hyperparameters
    grid_search = GridSearchCV(estimator=nb_model, param_grid=param_grid, cv=cv, scoring='f1', verbose=1)
    grid_search.fit(X_train_selected, y_train)

    # Best hyperparameters from grid search
    best_alpha = grid_search.best_params_['alpha']
    print(f'Best alpha: {best_alpha}')

    # Train the model with the best hyperparameter
    final_model = MultinomialNB(alpha=best_alpha)
    final_model.fit(X_train_selected, y_train)
    y_pred = final_model.predict(X_test_selected)

    # Calculating performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
    }
    if logs:
        # neg_class_prob_sorted = final_model.feature_log_prob_[0, :].argsort()[::-1]
        # pos_class_prob_sorted = final_model.feature_log_prob_[1, :].argsort()[::-1]

        # print(np.take(vectorizer.get_feature_names_out(), neg_class_prob_sorted[:5]))
        # print(np.take(vectorizer.get_feature_names_out(), pos_class_prob_sorted[:5]))
        # Ensure that feature_log_prob_ is correctly capturing the log probabilities
        feature_log_prob = final_model.feature_log_prob_
        
        # Ensure feature_names are extracted correctly
        feature_names = vectorizer.get_feature_names_out()

        # Calculate the difference in log probability between the two classes
        feature_importance = feature_log_prob[1, :] - feature_log_prob[0, :]

        # Get top 10 features for fake reviews (class 1) and genuine reviews (class 0)
        top_fake_indices = feature_importance.argsort()[-5:][::-1]
        top_genuine_indices = feature_importance.argsort()[:5]

        top_fake_features = [(feature_names[i], feature_importance[i]) for i in top_fake_indices]
        top_genuine_features = [(feature_names[i], feature_importance[i]) for i in top_genuine_indices]

        # Print top features indicating fake reviews
        print("\nTop 5 features indicating fake reviews:")
        for feature, score in top_fake_features:
            print(f'{feature}: {score:.4f}')

        # Print top features indicating genuine reviews
        print("\nTop 5 features indicating genuine reviews:")
        for feature, score in top_genuine_features:
            print(f'{feature}: {score:.4f}')


    return metrics


def logistic_reg(X_train, y_train, X_test, y_test, n_features=100, param_grid=None, type='unigram', remove_sparse=False):
    print('_____' * 20)
    print(f'Logistic Regression {type} Model')
    
    # Create matrix and transform data
    X_train_vec, X_test_vec, y_train, y_test, vectorizer = \
        create_matrix(X_train, X_test, y_train, y_test, type, remove_sparse)

    # Feature Selection using SelectKBest
    feature_selector = SelectKBest(chi2, k=n_features)
    X_train_selected = feature_selector.fit_transform(X_train_vec, y_train)
    X_test_selected = feature_selector.transform(X_test_vec)

    # Define the model
    lr_model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)

    # Set up the parameter grid if not provided
    if param_grid is None:
        param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 20, 40, 50, 70, 90, 100]}

    # Stratified K-Folds cross-validator
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Grid search for best hyperparameters
    grid_search = GridSearchCV(estimator=lr_model, param_grid=param_grid, cv=cv, scoring='f1')
    grid_search.fit(X_train_selected, y_train)

    # Best hyperparameters from grid search
    best_C = grid_search.best_params_['C']
    print(f'Best C: {best_C}')

    # Train the model with the best hyperparameter
    final_model = LogisticRegression(penalty='l1', C=best_C, solver='liblinear', max_iter=1000)
    final_model.fit(X_train_selected, y_train)

    # Step 3: Make predictions and evaluate the model
    y_pred = final_model.predict(X_test_selected)

    # Calculating performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
    }

    return metrics


def classification_tree(X_train, y_train, X_test, y_test, n_features=100, param_grid=None, type='unigram'):
    print('_____' * 20)
    print(f'Classification Tree {type} Model')
    
    # Vectorize the text data
    X_train_vectorized, X_test_vectorized, y_train, y_test, vectorizer = \
        create_matrix(X_train, X_test, y_train, y_test, type)

    # Define the model
    clf = DecisionTreeClassifier(random_state=42)

    # Set up the parameter grid if not provided
    if param_grid is None:
        param_grid = {
            'max_depth': [None, 5, 10, 15, 20, 25],
            'min_samples_split': [2, 5, 10, 15, 20]
        }

    # Stratified K-Folds cross-validator
    cv = StratifiedKFold(n_splits=5)

    # Grid search for best hyperparameters
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=cv, scoring='f1')
    grid_search.fit(X_train_vectorized, y_train)

    # Best hyperparameters from grid search
    best_params = grid_search.best_params_
    print(f'Best parameters: {best_params}')

    # Train the model with the best hyperparameters
    final_model = DecisionTreeClassifier(**best_params, random_state=42)
    final_model.fit(X_train_vectorized, y_train)

    # Step 3: Make predictions and evaluate the model
    y_pred = final_model.predict(X_test_vectorized)

    # Calculating performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
    }
    return metrics


def random_forest_model(X_train, y_train, X_test, y_test, n_features=100, param_grid=None, type='unigram'):
    print('_____' * 20)
    print(f'Random Forests {type} Model')
    
    # Vectorize the text data
    X_train_vectorized, X_test_vectorized, y_train, y_test, vectorizer = \
        create_matrix(X_train, X_test, y_train, y_test, type)
    
    # Feature Selection using SelectKBest
    feature_selector = SelectKBest(chi2, k=n_features)
    X_train_selected = feature_selector.fit_transform(X_train_vectorized, y_train)
    X_test_selected = feature_selector.transform(X_test_vectorized)
    
    # Define the model
    rf_model = RandomForestClassifier(random_state=42)

    # Set up the parameter grid if not provided
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10]
        }

    # Stratified K-Folds cross-validator
    cv = StratifiedKFold(n_splits=5)

    # Grid search for best hyperparameters
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=cv, scoring='f1')
    grid_search.fit(X_train_selected, y_train)

    # Best hyperparameters from grid search
    best_params = grid_search.best_params_
    print(f'Best parameters: {best_params}')

    # Train the model with the best hyperparameters
    final_model = RandomForestClassifier(**best_params, random_state=42)
    final_model.fit(X_train_selected, y_train)

    # Make predictions and evaluate the model
    y_pred = final_model.predict(X_test_selected)

    # Calculating performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
    }
    return metrics


run = [1, 0, 0, 0]

if run[0]:
    # Naive Bayes results
    # naive_bayes_unigram_results = naive_bayes(X_train, y_train, X_test, y_test, type='unigram', remove_sparse=False, logs=False)
    naive_bayes_unigram_results_sp_removed = naive_bayes(X_train, y_train, X_test, y_test, type='unigram', remove_sparse=True, logs=False)

    # naive_bayes_bigram_results = naive_bayes(X_train, y_train, X_test, y_test, type='bigram', remove_sparse=False, logs=False)
    naive_bayes_bigram_results_sp_removed = naive_bayes(X_train, y_train, X_test, y_test, type='bigram', remove_sparse=True, logs=False)

if run[1]:
    # Logistic Regression results
    logistic_reg_unigram_results = logistic_reg(X_train, y_train, X_test, y_test, type='unigram', remove_sparse=True)
    logistic_reg_bigram_results = logistic_reg(X_train, y_train, X_test, y_test, type='bigram', remove_sparse=True)

if run[2]:
    # Classification Tree results
    cls_tree_unigram_results = classification_tree(X_train, y_train, X_test, y_test, type='unigram')
    cls_tree_bigram_results = classification_tree(X_train, y_train, X_test, y_test, type='bigram')

if run[3]:
    # Random Forest results
    rf_unigram_results = random_forest_model(X_train, y_train, X_test, y_test, type='unigram')
    rf_bigram_results = random_forest_model(X_train, y_train, X_test, y_test, type='bigram')

if np.all(run):
    results = {
        'Naive Bayes Unigram': naive_bayes_unigram_results,
        'Naive Bayes Unigram Sparse Removed': naive_bayes_unigram_results_sp_removed,
        'Naive Bayes Bigram': naive_bayes_bigram_results,
        'Naive Bayes Bigram Sparse Removed': naive_bayes_bigram_results_sp_removed,
        'Logistic Regression Unigram': logistic_reg_unigram_results,
        'Logistic Regression Bigram': logistic_reg_bigram_results,
        'Classification Tree Unigram': cls_tree_unigram_results,
        'Classification Tree Bigram': cls_tree_bigram_results,
        'Random Forest Unigram': rf_unigram_results,
        'Random Forest Bigram': rf_bigram_results,
    }

    results_df = pd.DataFrame(results).T
    results_df.to_csv('data/results.csv')