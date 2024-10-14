import pandas as pd
import os
import re
from sklearn.feature_extraction.text import CountVectorizer 

# define all the directories that will be used
data_path = 'op_spam_v1.4'

negative = os.path.join(data_path, 'negative_polarity')
positive = os.path.join(data_path, 'positive_polarity')

negative_deceptive  = os.path.join(negative, 'deceptive_from_MTurk')
negative_truthful  = os.path.join(negative, 'truthful_from_Web')

positive_deceptive = os.path.join(positive, 'deceptive_from_MTurk')
positive_truthful = os.path.join(positive, 'truthful_from_TripAdvisor')

# define the names of the folfd that will be used for training and testing
folds = ['fold' + str(x) for x in range(1,6)]

# fetch all reviews in the different folders, put them in a list and in a df
# labeld according to their truthfulness
# we look through each folder, create a tuple of the review, truthfulness and positivity of the review and append them to the list
def fetch_reviews():
  
  train_reviews_x = [] # list of each tuple
  test_reviews_x = [] # the reviews we use for testing

  train_reviews_y = []
  test_reviews_y = []

  # all the different directories, each containing 5 folds
  dirs = [negative_truthful, positive_truthful, negative_deceptive, positive_deceptive]

  for fetch_dir in dirs:
    for fold in folds:
      fold_path = os.path.join(fetch_dir,fold)
      # print(f'{fetch_dir}/{fold}: {len(os.listdir(fold_path))} files')

      for filename in os.listdir(fold_path):
        file_path = os.path.join(fold_path, filename)
   
        # test in filepaths what type of review we have
        truthful = int('truthful' in file_path)
        positive = int('positive_polarity' in file_path)
        
        # open the content of each txt file and append to list
        # append to test reviews if fold5
        with open(file_path, 'r', encoding='utf-8') as file:
            doc_name = f'{fold}_{filename}_{str(positive)}{str(truthful)}'
            content = file.read()

            if not 'fold5' in file_path:
              train_reviews_x.append((file_path,content))
              train_reviews_y.append((file_path,truthful))
            else:
              test_reviews_x.append((file_path,content))
              test_reviews_y.append((file_path,truthful))
            
  return (train_reviews_x,test_reviews_x,train_reviews_y,test_reviews_y)


# This is the organized data, but not yet clean. We will clean the code with the CountVectorizer
(train_reviews_x,test_reviews_x,train_reviews_y,test_reviews_y) = fetch_reviews()

# function to pass along with the vectorizer
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[\d\W]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def create_dtm(reviews_x):
  vectorizer = CountVectorizer(stop_words='english',preprocessor=preprocess)
  
  doc_names = [t1 for (t1,_) in reviews_x]
  documents = [t2 for (_,t2) in reviews_x]
  
  dtm = vectorizer.fit_transform(documents)
  
  # Get feature names (words) for columns
  terms = vectorizer.get_feature_names_out()
  
  # Create the DataFrame with the DTM data and appropriate labels
  dtm_df = pd.DataFrame(dtm.toarray(), index=doc_names, columns=terms)

  return dtm_df


# because we seperated the train and test data too early, i need to join them and then seperate them again
dtm_full = create_dtm(train_reviews_x + test_reviews_x)
train_doc_names = [t1 for (t1,_) in train_reviews_x]
test_doc_names = [t1 for (t1,_) in test_reviews_x]

# These are the document term matrices! Use these as your x values.
train_dtm_df = dtm_full.loc[train_doc_names] # TRAIN X
test_dtm_df = dtm_full.loc[test_doc_names]   # TEST  X

# These are the y values
train_y_df = pd.DataFrame([t2 for (_,t2) in train_reviews_y],columns=['y']) # TRAIN Y
test_y_df = pd.DataFrame([t2 for (_,t2) in test_reviews_y],columns=['y'])   # TEST  Y