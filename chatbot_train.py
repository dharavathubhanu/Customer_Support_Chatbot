import json
import re
import nltk
import pickle
import numpy as np

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents JSON file
with open("intents.json", "r") as file:
    intents = json.load(file)

# Lists for training data
corpus = []
labels = []

# Clean, tokenize, lemmatize input patterns
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        text = re.sub(r'[^a-zA-Z ]', '', pattern.lower())  # Remove punctuation and lower text
        tokens = nltk.word_tokenize(text)
        lemma = " ".join([lemmatizer.lemmatize(token) for token in tokens])
        corpus.append(lemma)
        labels.append(intent["tag"])

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(labels)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X, y)

# Save the trained model and encoders
with open("intent_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print(" Model training complete and files saved.")
