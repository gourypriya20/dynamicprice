# Step 1: Import necessary libraries
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import re
import pickle

def car_prediction():
    # Define the predict_price function
    def predict_price(description, vectorizer, clothes_data, regressor):
        # Preprocess the new description
        tokens = word_tokenize(description.lower())
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]

        # Combine description features with new attributes
        description_features = vectorizer.transform([" ".join(tokens)]).toarray()[0]
        new_description_features = np.hstack([description_features, clothes_data[['km_covered', 'model_year']].iloc[0]])

        # Predict price using the model
        predicted_price = regressor.predict([new_description_features])[0]
        return predicted_price

    # Step 2: Load the HTML file
    file_path = 'templates/cars.html'
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()

    # Step 3: Parse the HTML content of the webpage using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Step 4: Extract product descriptions and prices from the webpage
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

    # Step 1: Load the dataset "car.csv" for price prediction
    clothes_data = pd.read_csv("model/car.csv")  # Assuming your CSV has columns "description", "price", "km_covered", and "model_year"

    # Step 2: Tokenize, lemmatize, and remove stopwords from the descriptions for price prediction
    tokenized_descriptions = []
    # Define lemmatizer and stop_words
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    for description in clothes_data["description"]:
        tokens = word_tokenize(description.lower())
        # Lemmatize tokens and remove stopwords
        tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalnum() and token.lower() not in stop_words]
        tokenized_descriptions.append(" ".join(tokens))

    # Step 3: Train a Random Forest Regressor for price prediction
    vectorizer = TfidfVectorizer()
    vectorizer.fit(tokenized_descriptions)
    all_features = np.hstack([vectorizer.transform(tokenized_descriptions).toarray(), clothes_data[['km_covered', 'model_year']]])

    regressor = RandomForestRegressor()
    regressor.fit(all_features, clothes_data["price"])

    with open('vectorizer_cars.pkl', 'wb') as vec_file:
        pickle.dump(vectorizer, vec_file)

    with open('regressor_cars.pkl', 'wb') as reg_file :
        pickle.dump(regressor, reg_file)

    # Step 4: Predict prices for new descriptions
    predicted_prices = []

    for description in descriptions:
        predicted_price = predict_price(description, vectorizer, clothes_data, regressor)
        predicted_prices.append(predicted_price)

    # Step 5: User input for price (assuming it's the actual prices extracted from the webpage)
    user_given_prices = prices

    # Step 6: Check for potential fraud based on predicted prices and user-given prices
    fraud_threshold = 0.2  # Adjust this value as needed (e.g., 20% above predicted price)

    for i, description in enumerate(descriptions):
        predicted_price = predicted_prices[i]
        user_given_price = user_given_prices[i]

        fraud_price = predicted_price * (1 + fraud_threshold)
        #print(f"User Input Price: Rs.{user_given_price:.2f}")
        #print(f"Predicted Price: Rs.{predicted_price:.2f}")

        if user_given_price > fraud_price:
            str =  "** FRAUD ALERT! User-given price exceeds threshold of predicted price for given discreption."
        else:
            str = "Price seems reasonable based on prediction for"
        return (predicted_price,str)

print(car_prediction())
