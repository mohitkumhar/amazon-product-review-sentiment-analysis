import pandas as pd
from bs4 import BeautifulSoup
import re

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import pickle
from fuzzywuzzy import fuzz

pd.options.mode.chained_assignment = None

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')



stop_words = set(stopwords.words('english'))
punc = string.punctuation
ws = WordNetLemmatizer()

with open ('model.pkl', 'rb') as file:
    model = pickle.load(file)
    
with open ('tfidf.pkl', 'rb') as file:
    tfdif = pickle.load(file)


def remove_url(text):
    if isinstance(text, str):  # Check if the input is a string
        pattern = re.compile(r'https?://\S+|www\.\S+')
        return pattern.sub(r'', text)
    return text  # If not a string, return it unchanged
    
def preprocessing(text):
    
    # removing urls
    text = remove_url(text)

    # remove html tags
    text = BeautifulSoup(text, 'html.parser')
    text = text.get_text()
    if not text.strip():  # Check if text is empty after stripping HTML
        return 'empty_text'

    # tokenizing the text
    text_list = word_tokenize(text)
    if not text_list:  # Check if text is empty after stripping HTML
        return 'empty_text'


    # lowering the words
    for i in range(len(text_list)):
        text_list[i] = text_list[i].lower().strip()

    # removing the stopwords
    filtered_words = []
    for word in text_list:
        if word not in stop_words:
            filtered_words.append(word)

    text_list = filtered_words

    # removing punctuation
    filtered_words = []
    for word in text_list:
        if word not in punc:
            filtered_words.append(word)
    text_list = filtered_words

    # stemming
    for i in range(len(text_list)):
        text_list[i] = text_list[i].replace('ing', '')
        text_list[i] = text_list[i].replace("'s", '')
        text_list[i] = text_list[i].replace("'re", '')
        text_list[i] = text_list[i].replace("'ve", '')
        text_list[i] = text_list[i].replace("'nt", '')
        text_list[i] = ws.lemmatize(text_list[i])

    final_text =  ' '.join(text_list)

    if not final_text.strip():
        return 'empty_text'
    return final_text




def fetch_fuzzy_features(row):
    q1 = str(row['char_len'])
    q2 = str(row['word_len'])

    fuzze_features = [0.0] * 4

    fuzze_features[0] = fuzz.QRatio(q1, q2)
    fuzze_features[1] = fuzz.partial_ratio(q1, q2)
    fuzze_features[2] = fuzz.token_sort_ratio(q1, q2)
    fuzze_features[3] = fuzz.token_set_ratio(q1, q2)

    return fuzze_features



def feature_engineering(row):
    row['char_len'] = row['Review'].str.len()
    row['word_len'] = row['Review'].apply(lambda x: len(x.split()))
    row['char_word_len_ratio'] = row['char_len'] / row['word_len']
    row['first_char_len'] = row['Review'].apply(lambda x: len(x.split()[0]))
    row['last_char_len'] = row['Review'].apply(lambda x: len(x.split()[-1]))
    fuzzy_features = row.apply(fetch_fuzzy_features, axis=1)
    row['fuzz_ratio'] = list(map(lambda x: x[0], fuzzy_features))
    row['fuzz_partial_ratio'] = list(map(lambda x: x[1], fuzzy_features))
    row['fuzz_token_sort_ratio'] = list(map(lambda x: x[2], fuzzy_features))
    row['fuzz_token_set_ratio'] = list(map(lambda x: x[3], fuzzy_features))
    return row
    
    


def predict_sentiment(text):
    text = preprocessing(text)
    text_df = pd.DataFrame([text], columns=['Review'])
    text_df = feature_engineering(text_df)
    
    review_df = text_df['Review']
    review_transform_df = pd.DataFrame(tfdif.transform(review_df).toarray())
    
    other_df = text_df.drop(columns=['Review'])
    
    review_transform_df = review_transform_df.reset_index(drop=True)
    other_df = other_df.reset_index(drop=True)
    
    final_df = pd.concat([other_df, review_transform_df], axis=1)
    
    final_df.columns = final_df.columns.astype(str)
    
    prediction = model.predict(final_df)
    
    return prediction
    


predict_sentiment("Best movie,  recommanded")



