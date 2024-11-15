# Amazon Review Sentiment Analysis

[**Demo Project**](https://amazon-review-sentimental-analysis.streamlit.app/)  
[**GitHub**](https://github.com/mohitkumhar/amazon-review-sentimental-analysis)

This project applies sentiment analysis on Amazon product reviews, classifying them as positive, negative, or neutral. By leveraging natural language processing (NLP) techniques and the XGBoost model, this project provides insights into customer satisfaction and feedback trends. This repository includes data preprocessing, feature extraction with TF-IDF, and sentiment classification using XGBoost.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Demo](#demo)
- [Portfolio](#portfolio)
- [Contributing](#contributing)

---

## Project Overview
Sentiment analysis is a powerful tool to understand customer feedback by analyzing textual data. This project specifically targets Amazon product reviews to classify customer feedback into positive, neutral, and negative sentiments. It can be useful for businesses to gain insights into their product reception and customer satisfaction.

---

## Features
- **Sentiment Classification**: Classifies reviews as positive, negative, or neutral.
- **NLP Processing**: Uses TF-IDF for feature extraction from textual data.
- **Modeling**: Employs XGBoost for accurate classification.
- **Data Visualization**: Provides visual insights into sentiment distribution.

---

## Dataset
The dataset for this project consists of Amazon product reviews, which can be downloaded from [Kaggle - Amazon Reviews Dataset](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews).  
The data is preprocessed to remove noise, tokenize text, and extract meaningful features.

---

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/mohitkumhar/amazon-review-sentimental-analysis.git
    cd amazon-review-sentimental-analysis
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download or prepare your dataset:
   - Download the dataset from [Kaggle - Amazon Reviews Dataset](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews).  
   - Save the dataset as `reviews.csv` in the root directory.

---

## Usage

Run the Streamlit application with:
   ```bash
   streamlit run app.py
   ```

This will start a local server. Open the provided URL in your browser to interact with the sentiment analysis app.

---

## Results
The XGBoost model achieves high accuracy in classifying Amazon product reviews as positive, neutral, or negative.  
The classification report and confusion matrix are available in the `results/` directory for deeper analysis.

---

## Demo
You can access the [Demo Project Here](https://amazon-review-sentimental-analysis.streamlit.app/).

---

## Portfolio
For more projects and contact information, visit my [Portfolio](https://mohits.live).

---

## Contributing
Feel free to fork the repository and make improvements. Pull requests are welcome!
