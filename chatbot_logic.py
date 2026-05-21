import json
import random
import re
import nltk
import pickle
import pandas as pd
from nltk.stem import WordNetLemmatizer
from ner_utils import extract_entities

# Download required tokenizer
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load ML components
model = pickle.load(open("intent_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))

# Load intents
with open("intents.json", "r") as file:
    data = json.load(file)

# Optional FAQ CSV
def load_faq_responses():
    try:
        faq_df = pd.read_csv("faq.csv")
        faq_dict = {}
        for _, row in faq_df.iterrows():
            tag = row["tag"]
            faq_dict.setdefault(tag, []).append(row["answer"])
        return faq_dict
    except FileNotFoundError:
        return {}

faq_responses = load_faq_responses()

# Preprocess input
def preprocess_input(text):
    text = re.sub(r'[^a-zA-Z ]', '', text.lower())
    tokens = nltk.word_tokenize(text)
    return " ".join([lemmatizer.lemmatize(word) for word in tokens])

# Predict intent
def predict_intent(text):
    lemma = preprocess_input(text)
    print("Preprocessed input:", lemma)
    X = vectorizer.transform([lemma])
    proba = model.predict_proba(X)[0]
    print("Confidence scores:", proba)
    max_index = proba.argmax()
    confidence = proba[max_index]
    tag = encoder.inverse_transform([max_index])[0]
    print("Predicted tag:", tag, "| Confidence:", confidence)
    return tag, confidence

# Get response from tag
def get_response(tag):
    if tag in faq_responses:
        return random.choice(faq_responses[tag])
    for intent in data["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "I'm not sure how to help with that."

# Main response logic
def chatbot_response(user_input, previous_intent=None):
    tag, confidence = predict_intent(user_input)

    # Handle greeting directly
    greeting_inputs = ["hi", "hello", "hey", "good morning", "good evening"]
    if user_input.strip().lower() in greeting_inputs:
        return random.choice(["Hello! How can I help you?", "Hi there!", "Greetings!"]), "greeting"

    if confidence < 0.2:
        return "I'm not sure what you mean. Can you please rephrase?", "clarification"

    # Extract entities (order ID, etc.)
    entities = extract_entities(user_input)

    # Refund intent
    if tag == "refund":
        order_id_match = re.search(r"\b\d{4,10}\b", user_input)
        if order_id_match:
            order_id = order_id_match.group()
            response = f"Thank you! Your refund request for order ID {order_id} has been received and is being processed."
            tag = "refund_completed"
        else:
            response = "Sure, I can help you with a refund. Can you please provide the order ID?"

    elif previous_intent == "refund":
        order_id_match = re.search(r"\b\d{4,10}\b", user_input)
        if order_id_match:
            order_id = order_id_match.group()
            response = f"Thank you! Your refund request for order ID {order_id} has been received and is being processed."
            tag = "refund_completed"
        else:
            response = "I couldn't find a valid order ID. Please provide a numeric ID like 12345."

    elif tag == "working_hours":
        response = "We are open from 9 AM to 6 PM, Monday to Friday."

    elif tag == "order_status":
        response = "Could you please share your order ID so I can check the status?"

    elif tag == "greeting":
        response = random.choice(["Hello! How can I help you?", "Hi there!", "Greetings!"])

    elif tag == "goodbye":
        response = random.choice(["Goodbye! Have a great day!", "Bye! Come back anytime."])

    elif tag == "thanks":
        response = random.choice(["You're welcome!", "No problem!", "Glad I could help."])

    else:
        response = get_response(tag)

    # Append entity debug info
    if entities:
        response += f" (I noticed: {entities})"

    return response, tag
