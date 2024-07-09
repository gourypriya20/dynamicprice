# Step 1: Import necessary libraries
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from textblob import TextBlob
import numpy as np
import re

# Define the predict_price function
def predict_price(description, vectorizer, clothes_data, regressor):
    # Preprocess the new description
    tokens = word_tokenize(description.lower())
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]

    # Attempt to extract age feature from description (optional)
    age = 0  # Default age if not found
    age_pattern = r"\d+ year(?:s?) old"  # Example regex for "X years old"
    match = re.search(age_pattern, description, flags=re.IGNORECASE)
    if match:
        age = int(match.group(0).split()[0])

    # Combine description features and age into a list
    description_features = vectorizer.transform([" ".join(tokens)]).toarray()[0]
    new_description_features = np.hstack([description_features, age])

    # Predict price using the model
    predicted_price = regressor.predict([new_description_features])[0]
    return predicted_price

def shoe():
    # Step 1: Load the HTML file
    file_path = 'templates/shoes.html'
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()

    # Step 2: Parse the HTML content of the webpage using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Step 3: Extract product descriptions and prices from the webpage
    descriptions = []
    prices = []

    for product in soup.find_all('div', class_='single-pro-details'):
        description_element = product.find('span', class_='desc')  # Try finding with 'desc' class first
        if not description_element:
            description_element = product.find('p')  # If not found, try finding a paragraph element
        if description_element:
            description = description_element.text.strip()
            descriptions.append(description)

            # Extract price
            price_element = product.find('h2')  # Assuming the price is contained within <h2> tags
            if price_element:
                price_text = price_element.text.strip()
                # Extract numerical part of the price
                price = float(re.search(r'\d+(\.\d+)?', price_text).group())
                prices.append(price)
            else:
                print("Price not found for a product.")
        else:
            print("Product description not found for a product.")

    # Step 4: Tokenize, lemmatize, and remove stopwords from the descriptions
    tokenized_descriptions = []
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    for description in descriptions:
        tokens = word_tokenize(description.lower())
        # Lemmatize tokens and remove stopwords
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token.lower() not in stop_words]
        tokenized_descriptions.append(" ".join(tokens))

    # Step 5: Load sentiment data and train a sentiment classification model
    sentiment_data = pd.read_csv("model/posineg.csv")  # Assuming your CSV has columns "word" and "sentiment"
    words = sentiment_data["word"].tolist()
    labels = sentiment_data["sentiment"].tolist()

    vectorizer = TfidfVectorizer()
    word_features = vectorizer.fit_transform(words)

    classifier = LogisticRegression()
    classifier.fit(word_features, labels)

    # Step 6: Predict sentiment for the product descriptions
    predicted_sentiments = []

    for tokens in tokenized_descriptions:
        description = " ".join(tokens)
        new_description_features = vectorizer.transform([description])
        predicted_sentiment = classifier.predict(new_description_features)[0]
        predicted_sentiments.append(predicted_sentiment)

    # Step 7: Load the dataset "shoe.csv" for price prediction
    clothes_data = pd.read_csv("model/shoe_and_bag.csv")  # Assuming your CSV has columns "description" and "price"

    # Step 8: Tokenize, lemmatize, and remove stopwords from the descriptions for price prediction
    tokenized_descriptions = []

    for description in clothes_data["description"]:
        tokens = word_tokenize(description.lower())
        # Lemmatize tokens and remove stopwords
        tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalnum() and token.lower() not in stop_words]
        tokenized_descriptions.append(" ".join(tokens))

    # Step 9: Train a Random Forest Regressor for price prediction
    vectorizer = TfidfVectorizer()
    vectorizer.fit(tokenized_descriptions)

    all_features = np.hstack([vectorizer.transform(tokenized_descriptions).toarray(), np.array(clothes_data["years_used"]).reshape(-1, 1)])

    regressor = RandomForestRegressor()
    regressor.fit(all_features, clothes_data["price"])

    # Step 10: Predict prices for new descriptions
    predicted_prices = []

    for description in descriptions:
        predicted_price = predict_price(description, vectorizer, clothes_data, regressor)
        predicted_prices.append(predicted_price)

    # Step 11: User input for price (assuming it's the actual prices extracted from the webpage)
    user_given_prices = prices

    # Step 12: Check for potential fraud based on predicted prices and user-given prices
    fraud_threshold = 0.2  # Adjust this value as needed (e.g., 20% above predicted price)

    for i, description in enumerate(descriptions):
            predicted_price = predicted_prices[i]
            user_given_price = user_given_prices[i]

            fraud_price = predicted_price * (1 + fraud_threshold)

            if user_given_price > fraud_price:
                str = "** FRAUD ALERT! User-given price exceeds threshold of predicted price"
                
            else:
                str = "Price seems reasonable based on prediction."
    return(predicted_prices[i], str)

print(shoe())