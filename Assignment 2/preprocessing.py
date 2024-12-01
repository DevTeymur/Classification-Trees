import pandas as pd
import string
import re
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation)) # Remove punctuation
    text = ' '.join(word for word in text.split() if word not in stop_words) # Remove stopwords
    text = re.sub(r'\d+', '', text) # Remove numbers
    text = ' '.join(text.split()) # Extra whitespaces
    return text

df = pd.read_csv('data/raw_data.csv')

# Assuming df is your dataframe with a 'text' column
df['cleaned_text'] = df['text'].apply(clean_text)

# Display the first few rows of the cleaned data
print(df[['text', 'cleaned_text']].head())

clean_data = df[['cleaned_text', 'class']]
clean_data.to_csv('data/cleaned_data.csv', index=False)
