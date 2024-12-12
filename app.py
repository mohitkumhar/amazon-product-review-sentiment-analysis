import requests
from bs4 import BeautifulSoup
import streamlit as st
import pickle
from fuzzywuzzy import fuzz
import numpy as np
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import string

nltk.download('stopwords')


sentiment_model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))


with open('keyword extraction/count_vectorizer.pkl', 'rb') as file:
    keyword_cv = pickle.load(file, encoding='utf-8')


with open('keyword extraction/feature_names.pkl', 'rb') as file:
    keyword_feature_names = pickle.load(file, encoding='utf-8')


with open('keyword extraction/tfidf_transformer.pkl', 'rb') as file:
    keyword_tfidf_transformer = pickle.load(file, encoding='utf-8')


ps = PorterStemmer()
ws = WordNetLemmatizer()

stop_words = set(stopwords.words('english'))
punc = string.punctuation


def remove_url(text):
    if isinstance(text, str):  # Check if the input is a string
        pattern = re.compile(r'https?://\S+|www\.\S+')
        return pattern.sub(r'', text)
    return text  # If not a string, return it unchanged
    
def preprocessing(text):
    
    # removing urls
    text = re.sub(r'Read more', '', text)
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

# Function to scrape reviews using BeautifulSoup
def scrape_reviews(url, review_type, sort_by, num_reviews):
    reviews = []
    page_number = 1
    
    while len(reviews) < num_reviews:
        # Adjust URL with filters and pagination
        filtered_url = f"{url}?reviewerType={review_type}&sortBy={sort_by}&pageNumber={page_number}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36"
        }
        response = requests.get(filtered_url, headers=headers)

        # Check if the page is fetched successfully
        if response.status_code != 200:
            break
        
        # Parse the page using BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Find all review elements
        review_elements = soup.find_all("span", {"data-hook": "review-body"})
        
        # Extract reviews
        for element in review_elements:
            reviews.append(element.get_text().strip())
            if len(reviews) >= num_reviews:
                break

        # Move to the next page
        page_number += 1
        
        # Check if there is no next page
        if not soup.find("a", {"data-hook": "see-all-reviews-link-foot"}):
            break

    return reviews

    


# Function to predict sentiment score for reviews
def calculate_overall_sentiment(reviews):
    # Assuming the model outputs probabilities or scores for sentiment
        
    df = pd.DataFrame(reviews, columns=["Review"])
    
    df['Review'] = df['Review'].apply(preprocessing)
    
    reviews = df['Review'].tolist()
    
    review_transform = tfidf.transform(reviews).toarray()
    
    review_temp_df = pd.DataFrame(review_transform)
        
    df['char_len'] = df['Review'].str.len()
    df['word_len'] = df['Review'].apply(lambda x: len(x.split()))
    df['char_word_len_ratio'] = df['char_len'] / df['word_len']
    df['first_char_len'] = df['Review'].apply(lambda x: len(x.split()[0]))
    df['last_char_len'] = df['Review'].apply(lambda x: len(x.split()[-1]))
    fuzzy_features = df.apply(fetch_fuzzy_features, axis=1)
    df['fuzz_ratio'] = list(map(lambda x: x[0], fuzzy_features))
    df['fuzz_partial_ratio'] = list(map(lambda x: x[1], fuzzy_features))
    df['fuzz_token_sort_ratio'] = list(map(lambda x: x[2], fuzzy_features))
    df['fuzz_token_set_ratio'] = list(map(lambda x: x[3], fuzzy_features))
    
    other_temp_df = df.drop(columns = ['Review'])
    
    other_temp_df = other_temp_df.reset_index(drop=True)
    review_temp_df = review_temp_df.reset_index(drop=True)
    
    final_df = pd.concat([other_temp_df, review_temp_df], axis=1)


    
    # Get predictions from the sentiment model
    predictions = sentiment_model.predict(final_df)
    
    # Calculate the overall sentiment as the mode (most frequent sentiment)
    overall_sentiment = np.bincount(predictions).argmax()
    
    return overall_sentiment


def get_keyword_score(words, topN=10):
    
    words = keyword_tfidf_transformer.transform(keyword_cv.transform([words]))

    words = words.tocoo()
    
    tuples = zip(words.col, words.data)
    sorted_items = sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
    sorted_items = sorted_items[:topN]
    
    score_val = []
    features_val = []
    
    for idx, score in sorted_items:
        score_val.append(round(score, 3))
        features_val.append(keyword_feature_names[idx])
    
    results = {}
    for idx in range(len(features_val)):
        results[features_val[idx]] = score_val[idx]
    
    return results
        
    

# Streamlit UI
st.title("Amazon Review Sentiment Analyzer")

# User inputs
url = st.text_input("Enter the Amazon product review URL:")
# url = 'https://www.amazon.in/Refurbished-Lenovo-ThinkPad-Bluetooth-Graphics/dp/B0DMTPY8PJ/'
review_type = st.selectbox("Review Type:", ("all_reviews", "avp_only_reviews"))
sort_by = st.selectbox("Sort By:", ("helpful", "recent"))
num_reviews = st.number_input("Number of Reviews:", min_value=1, max_value=100, value=10)


if st.button("Analyze Reviews"):
    if url:
        # Scrape reviews
        st.write("Fetching reviews from Amazon...")
        reviews = scrape_reviews(url, review_type, sort_by, num_reviews)
        
        if reviews:
            st.write(f"Fetched {len(reviews)} reviews.")
            
            # Combine all reviews into one text
            combined_reviews = " ".join(reviews)
            combined_reviews = re.sub(r'Read more', '', combined_reviews)
            reviews_keyword_score = get_keyword_score(combined_reviews)
            
            # Calculate overall sentiment
            overall_sentiment = calculate_overall_sentiment(reviews)
            
            # Display results
            st.write("**Common Review:**")
            
            # Add a checkbox to let the user decide whether to display the table
            # show_table = st.checkbox("Show Keyword Scores")
            
            # if show_table:
                # Convert the dictionary to a DataFrame
            df = pd.DataFrame(list(reviews_keyword_score.items()), columns=["Word", "Score"])
            
            # Display the DataFrame as a table
            st.table(df)
            
            st.write("**Overall Sentiment:**")
            
            if overall_sentiment == 1:
                overall_sentiment = "Positive"
            else:
                overall_sentiment = "Negative"
            st.write(f"The overall sentiment is **{overall_sentiment}**.")
        else:
            st.write("No reviews found.")
    else:
        st.write("Please enter a valid URL.")