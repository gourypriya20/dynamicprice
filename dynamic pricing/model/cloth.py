# Step 1: Import necessary libraries
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
import pickle

def predict_price(description, vectorizer, regressor):
    tokens = word_tokenize(description.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]
    age = 0 
    age_pattern = r"\d+ year(?:s?) old"
    match = re.search(age_pattern, description, flags=re.IGNORECASE)
    if match:
        age = int(match.group(0).split()[0])
    # Combine description features and age into a list
    description_features = vectorizer.transform([" ".join(tokens)]).toarray()[0]
    new_description_features = np.hstack([description_features, age])
    # Predict price using the model
    predicted_price = regressor.predict([new_description_features])[0]
    return predicted_price

# Step 1: Load the HTML file
file_path = 'templates/cloth.html'

with open(file_path, 'r', encoding='utf-8') as file:
    html_content = file.read()
# Step 2: Parse the HTML content of the webpage using BeautifulSoup
soup = BeautifulSoup(html_content, 'html.parser')

# Step 3: Extract product descriptions from the webpage
descriptions = []
for product in soup.find_all('div', class_='single-pro-details'):
    description_element = product.find('span', class_='desc')  # Try finding with 'desc' class first
    if not description_element:
        description_element = product.find('p')  # If not found, try finding a paragraph element
    if description_element:
        description = description_element.text.strip()
        descriptions.append(description)
    else:
        print("Product description not found for a product.")

# Step 4: Tokenize, lemmatize, and remove stopwords from the descriptions
tokenized_descriptions = []
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

for description in descriptions:
    tokens = word_tokenize(description)
    # Lemmatize tokens and remove stopwords
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalnum() and token.lower() not in stop_words]
    tokenized_descriptions.append(" ".join(tokens))

# Step 5: Load sentiment data
sentiment_data = pd.read_csv("model/posineg.csv")  
words = sentiment_data["word"].tolist()
labels = sentiment_data["sentiment"].tolist()

# Step 6: Vectorize the words (convert them to numerical features)
vectorizer = TfidfVectorizer()
word_features = vectorizer.fit_transform(words)

# Open the file in write binary mode ("wb")
with open("vectorizer_cloth.pkl", "wb") as file:
  pickle.dump(vectorizer, file)

# Step 7: Train a machine learning model for sentiment classification
classifier = LogisticRegression()
classifier.fit(word_features, labels)

# Open the file in write binary mode ("wb")
with open("classifier_cloth.pkl", "wb") as file:
  pickle.dump(classifier, file)

# Step 8: Predict sentiment for new descriptions (using the model)
predicted_sentiments = []
for tokens in tokenized_descriptions:
    # Combine tokens back to a string
    description = " ".join(tokens)
    new_description_features = vectorizer.transform([description])
    predicted_sentiment = classifier.predict(new_description_features)[0]
    predicted_sentiments.append(predicted_sentiment)

# Step 9: Load the dataset "clothes.csv" (Price prediction)
clothes_data = pd.read_csv("model/cloth.csv")  # Assuming your CSV has columns "description" and "price"

# Step 10: Tokenize, lemmatize, and remove stopwords from the descriptions (Price prediction)
tokenized_descriptions = []
for description in clothes_data["description"]:
    tokens = word_tokenize(description.lower())
    # Lemmatize tokens and remove stopwords
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalnum() and token.lower() not in stop_words]
    tokenized_descriptions.append(" ".join(tokens))

# Step 11: Vectorize the descriptions for price prediction
vectorizer = TfidfVectorizer()
vectorizer.fit(tokenized_descriptions)  # Reuse the same vectorizer

# Step 12: Combine description features and age feature
all_features = np.hstack([vectorizer.transform(tokenized_descriptions).toarray(), np.array(clothes_data["years_used"]).reshape(-1, 1)])

# Step 13: Train a machine learning model for price prediction (Random Forest Regressor)
regressor = RandomForestRegressor()
regressor.fit(all_features, clothes_data["price"])

# Open the file in write binary mode ("wb")
with open("regressor_cloth.pkl", "wb") as file:
  pickle.dump(regressor, file) 

# Step 14: Predict prices for new descriptions
predicted_prices = []
for description in descriptions:
    predicted_price = predict_price(description, vectorizer, regressor)
    predicted_prices.append(predicted_price)

# Step 15: Access existing price data
actual_prices = clothes_data["price"].to_numpy()

# Step 16: Percentile Threshold (choose your desired percentile)
percentile_threshold = np.percentile(actual_prices, 99)  # Example: 99th percentile

# Step 17: Identify outliers based on percentile
percentile_outliers = [i for i, price in enumerate(predicted_prices) if price > percentile_threshold]
percentile_fraud = ["Potential Overpricing"] * len(percentile_outliers)  # Classify as potential overpricing

# Step 18: Filter data within percentile range (for standard deviation)
filtered_predicted_prices = [price for i, price in enumerate(predicted_prices) if price <= percentile_threshold]

# Step 19: Standard deviation and limits (assuming 2 standard deviations)
mean_filtered_prices = np.mean(filtered_predicted_prices)
std_filtered_prices = np.std(filtered_predicted_prices)
upper_limit = mean_filtered_prices + (2 * std_filtered_prices)
lower_limit = mean_filtered_prices - (2 * std_filtered_prices)


# Step 20: Identify outliers within the range based on standard deviation
std_outliers = [i for i, price in enumerate(filtered_predicted_prices) 
                if price > upper_limit or price < lower_limit]
std_fraud = ["Unusual Price Variation"] * len(std_outliers)

# Step 21: Combine outliers and fraud classifications from both methods
all_outlier_indices = list(set(percentile_outliers + std_outliers))
all_fraud_classifications = percentile_fraud + std_fraud


def print_classifications(descriptions, predicted_prices, all_outlier_indices, all_fraud_classifications):
  for i, description in enumerate(descriptions):
    text = TextBlob(description)
    sentiment = text.sentiment
    fraud_classification = "Reasonable price"  
    if i in all_outlier_indices:
      fraud_classification = all_fraud_classifications[all_outlier_indices.index(i)]
      
    with open("fraud_classifications_cloth.pkl", "wb") as file:
        pickle.dump(all_fraud_classifications, file)
    return predicted_prices[i],fraud_classification

def read():
    return print_classifications(descriptions, predicted_prices, all_outlier_indices, all_fraud_classifications)




