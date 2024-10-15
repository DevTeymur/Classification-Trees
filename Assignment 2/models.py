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


def naive_bayes(X_train, y_train, X_test, y_test, n_features=1000, type='uigram'):
    print('_____'*20)
    print(f'Naive Bayes {type} Model')
    # Vectorization unigram
    vectorizer = CountVectorizer(ngram_range=(1, 2))  if type == 'bigram' else CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    # Feature Selection using SelectKBest
    # Selecting the top 1000 features based on chi-squared test
    feature_selector = SelectKBest(chi2, k=n_features)
    X_train_selected = feature_selector.fit_transform(X_train_vectorized, y_train)
    X_test_selected = feature_selector.transform(X_test_vectorized)

    # Hyperparameter tuning using cross-validation
    nb_model = MultinomialNB()
    param_grid = {'alpha': np.arange(0.1, 5.1, 0.1)}
    cv = StratifiedKFold(n_splits=10)

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

    # Feature importance based on log probabilities
    feature_log_prob = final_model.feature_log_prob_  # Log probabilities of features
    feature_names = vectorizer.get_feature_names_out()

    # Difference in log probability between the two classes
    feature_importance = feature_log_prob[1, :] - feature_log_prob[0, :]

    # Get top 10 features for fake reviews (class 1) and genuine reviews (class 0)
    top_fake_indices = feature_importance.argsort()[-10:][::-1]
    top_genuine_indices = feature_importance.argsort()[:10]

    top_fake_features = [(feature_names[i], feature_importance[i]) for i in top_fake_indices]
    top_genuine_features = [(feature_names[i], feature_importance[i]) for i in top_genuine_indices]

    print("\nTop 10 features indicating fake reviews:")
    for feature, score in top_fake_features:
        print(f'{feature}: {score:.4f}')

    print("\nTop 10 features indicating genuine reviews:")
    for feature, score in top_genuine_features:
        print(f'{feature}: {score:.4f}')

    return metrics


def logistic_reg(X_train, y_train, X_test, y_test, n_features=1000, param_grid=None, type='unigram'):
    print('_____'*20)
    print(f'Logistic Regression {type} Model')
    vectorizer = CountVectorizer() if type == 'unigram' else CountVectorizer(ngram_range=(1, 2))
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    # Feature Selection using SelectKBest
    feature_selector = SelectKBest(chi2, k=n_features)
    X_train_selected = feature_selector.fit_transform(X_train_vectorized, y_train)
    X_test_selected = feature_selector.transform(X_test_vectorized)

    # Define the model
    lr_model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)

    # Set up the parameter grid if not provided
    if param_grid is None:
        param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 20, 40, 50, 70, 90, 100]}

    # Stratified K-Folds cross-validator
    cv = StratifiedKFold(n_splits=10)

    # Grid search for best hyperparameters
    grid_search = GridSearchCV(estimator=lr_model, param_grid=param_grid, cv=cv, scoring='f1')
    grid_search.fit(X_train_selected, y_train)

    # Best hyperparameters from grid search
    best_C = grid_search.best_params_['C']
    print(f'Best C: {best_C}')

    # Train the model with the best hyperparameter
    final_model = LogisticRegression(penalty='l1', C=best_C, solver='liblinear')
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



def classification_tree(X_train, y_train, X_test, y_test, n_features=1000, param_grid=None, type='unigram'):
    print('_____' * 20)
    print(f'Classification Tree {type} Model')
    
    # Vectorize the text data
    vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=n_features) if type == 'bigram' else CountVectorizer(max_features=n_features)
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    # Define the model
    clf = DecisionTreeClassifier(random_state=42)

    # Set up the parameter grid if not provided
    if param_grid is None:
        param_grid = {
            'max_depth': [None, 5, 10, 15, 20, 25],
            'min_samples_split': [2, 5, 10, 15, 20]
        }

    # Stratified K-Folds cross-validator
    cv = StratifiedKFold(n_splits=10)

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


def random_forest_model(X_train, y_train, X_test, y_test, n_features=1000, param_grid=None, type='unigram'):
    print('_____' * 20)
    print(f'Random Forests {type} Model')
    
    # Vectorize the text data
    vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=n_features) if type == 'bigram' else CountVectorizer(max_features=n_features)
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

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
    cv = StratifiedKFold(n_splits=10)

    # Grid search for best hyperparameters
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=cv, scoring='f1')
    grid_search.fit(X_train_vectorized, y_train)

    # Best hyperparameters from grid search
    best_params = grid_search.best_params_
    print(f'Best parameters: {best_params}')

    # Train the model with the best hyperparameters
    final_model = RandomForestClassifier(**best_params, random_state=42)
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

naive_bayes_unigram_results = naive_bayes(X_train, y_train, X_test, y_test, type='unigram')
naive_bayes_bigram_results = naive_bayes(X_train, y_train, X_test, y_test, type='bigram')
exit()
logistic_reg_unigram_results = logistic_reg(X_train, y_train, X_test, y_test, type='unigram')
logistic_reg_bigram_results = logistic_reg(X_train, y_train, X_test, y_test, type='bigram')
cls_tree_unigram_results = classification_tree(X_train, y_train, X_test, y_test, type='unigram')
cls_tree_bigram_results = classification_tree(X_train, y_train, X_test, y_test, type='bigram')
rf_unigram_results = random_forest_model(X_train, y_train, X_test, y_test, type='unigram')
rf_bigram_results = random_forest_model(X_train, y_train, X_test, y_test, type='bigram')

# Gather all results in csv file
results = {
    'Naive Bayes Unigram': naive_bayes_unigram_results,
    'Naive Bayes Bigram': naive_bayes_bigram_results,
    'Logistic Regression Unigram': logistic_reg_unigram_results,
    'Logistic Regression Bigram': logistic_reg_bigram_results,
    'Classification Tree Unigram': cls_tree_unigram_results,
    'Classification Tree Bigram': cls_tree_bigram_results,
    'Random Forest Unigram': rf_unigram_results,
    'Random Forest Bigram': rf_bigram_results,
}

results_df = pd.DataFrame(results).T
results_df.to_csv('data/results.csv')