import os
import pandas as pd

def read_reviews(base_path):
    data = []
    
    for review_type in ['deceptive_from_MTurk', 'truthful_from_Web']:
        review_path = os.path.join(base_path, 'negative_polarity', review_type)
        review_class = 0 if 'deceptive' in review_type else 1
        print(review_path, review_class)

        for fold in os.listdir(review_path):
            fold_path = os.path.join(review_path, fold)

            if os.path.isdir(fold_path):
                print(fold_path)
                
                for filename in os.listdir(fold_path):
                    if filename.endswith(".txt"):
                        file_path = os.path.join(fold_path, filename)
                        
                        with open(file_path, 'r', encoding='utf-8') as file:
                            text = file.read().strip()
                            data.append({'text': text, 'class': review_class})
    
    return pd.DataFrame(data)

base_path = '.'  # Set your base directory here
df = read_reviews(base_path)
print(df.head())  # Display the first few rows of the DataFrame

df.to_csv('data/raw_data.csv', index=False)
