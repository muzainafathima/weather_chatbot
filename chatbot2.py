import re
import random
import requests
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('wordnet')

# API Config
API_KEY = ''  # Replace with your OpenWeatherMap API key
BASE_URL = 'http://api.openweathermap.org/data/2.5/weather'

# Training Data
training_data = [
    {"intent": "greeting", "patterns": ["hello", "hi", "hey", "good morning", "good evening"], "responses": ["Hello!", "Hi there!", "Hey! How can I assist you? Here are some things you can ask:\n- Check the weather\n- Say goodbye\n- Ask for help"]},
    {"intent": "goodbye", "patterns": ["bye", "goodbye", "see you", "take care"], "responses": ["Goodbye!", "See you soon!", "Take care!"]},
    {"intent": "thanks", "patterns": ["thanks", "thank you", "appreciate it"], "responses": ["You're welcome!", "Happy to help!", "No problem!"]},
    {"intent": "help", "patterns": ["help", "assist me", "support"], "responses": ["Sure, I can help! Here are some things you can ask:\n- 'What's the weather in [city]?''Hi' or 'Hello','Bye' or 'Goodbye'"]},
    {"intent": "weather", "patterns": ["weather in", "what's the weather", "how's the weather in"], "responses": ["Sure! Please provide the city name to get the current weather."]},
    {"intent": "unknown", "patterns": [], "responses": ["I'm not sure I understand. Could you rephrase that? Here's what I can do:\n- Provide weather updates\n- Greet you\n- Say goodbye\n- Offer help"]}
]

# Preprocess Function
def preprocess(text):
    text = text.lower()
    words = nltk.word_tokenize(text)
    return ' '.join(words)

# Vectorize Data
vectorizer = CountVectorizer()
corpus = [preprocess(pattern) for data in training_data for pattern in data['patterns']]
vectorizer.fit(corpus)

# Find User Intent
def find_intent(user_input):
    user_input = preprocess(user_input)
    user_vec = vectorizer.transform([user_input])
    corpus_vec = vectorizer.transform(corpus)
    similarities = cosine_similarity(user_vec, corpus_vec)
    index = np.argmax(similarities)
    intent = None
    
    cumulative_index = 0
    for data in training_data:
        if index < cumulative_index + len(data['patterns']):
            intent = data
            break
        cumulative_index += len(data['patterns'])
    return intent

# Get Weather Data
def get_weather(city):
    params = {'q': city, 'appid': API_KEY, 'units': 'metric'}
    try:
        response = requests.get(BASE_URL, params=params)
        data = response.json()
        if data["cod"] == 200:
            weather = data["weather"][0]["description"]
            temp = data["main"]["temp"]
            return f"The weather in {city} is currently {weather} with a temperature of {temp}°C."
        else:
            return "Sorry, I couldn't find weather information for that city."
    except Exception as e:
        return "There was an error fetching the weather data."

# Chatbot Response Logic
def chatbot_response(user_input):
    user_input = preprocess(user_input)
    if "weather" in user_input and "in" in user_input:
        city = user_input.split("in")[-1].strip()
        return get_weather(city)
    
    intent = find_intent(user_input)
    if intent and intent["responses"]:
        return random.choice(intent["responses"])
    else:
        return random.choice(training_data[-1]["responses"])

# Run Chatbot
print("Chatbot: Hi! Ask me about the weather or type 'bye' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ['bye', 'exit']:
        print("Chatbot: Goodbye!")
        break
    print("Chatbot:", chatbot_response(user_input))
