import streamlit as st
from helper import predict_sentiment

def main():
    st.title("Amazon Reviews Sentiment Analysis")

    user_input = st.text_input("Enter a sentence:")

    if st.button("Predict"):
        if len(user_input.split()) < 3:
            st.error("Please enter at least 3 words.")
        else:
            prediction = predict_sentiment(user_input)
            if prediction[0] == 1:
                st.success(f"This Gives Positive Sentiment")
            
            else:
                st.success(f"This Gives Negative Sentiment")

if __name__ == "__main__":
    main()