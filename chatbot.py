import requests
import json
import random
import numpy as np
import pickle
import nltk
import threading
import time
from keras.models import load_model
from typing import Final
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Download necessary NLTK resources
nltk.download('punkt')

# Initialize the stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Global variables for model and data
model = None
intents = None
words = None
classes = None

def load_data():
    global model, intents, words, classes
    model = load_model('model.h5')
    intents = json.loads(open('data.json').read())
    words = pickle.load(open('texts.pkl', 'rb'))
    classes = pickle.load(open('labels.pkl', 'rb'))
    print(f"Data reloaded. Vocabulary size: {len(words)}")

def data_refresh_thread():
    while True:
        load_data()
        time.sleep(5)

# Start the data refresh thread
thread = threading.Thread(target=data_refresh_thread)
thread.daemon = True
thread.start()

TOKEN: Final = 
BOT_USERNAME: Final = 

def clean_up_sentence(sentence):
    # Tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # Stem each word - create short form for word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    # Tokenize and stem the pattern
    sentence_words = clean_up_sentence(sentence)
    # Bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)  
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s: 
                # Assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    # Ensure the bag length matches the expected input size
    if len(bag) != len(words):
        print(f"Warning: Bag length {len(bag)} does not match words length {len(words)}")
    return np.array(bag)

def predict_class(sentence, model):
    # Filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # Sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Commands
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Hello! Thanks for chatting with me!')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('I am here! Please type something so I can respond!')

async def custom_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('This is a custom command!')

# async def link_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     await update.message.reply_text('https://www.youtube.com/watch?v=vZtm1wuA2yc')

# Responses
def handle_response(text: str, username: str) -> str:
    ints = predict_class(text, model)
    if not ints or len(ints) == 0:
        return "Maaf saya tidak tau"
    
    top_intent = ints[0]
    tag = top_intent['intent']
    confidence = float(top_intent['probability'])
    print(f"Confidence: {confidence}") # Added print statement for confidence
    
    # if confidence < 0.5:  # Adjust the threshold as needed
    #     no_answer(text, username)
    #     return "Maaf saya tidak tau"
    
    if tag == 'noanswer':
        no_answer(text, username)
    
    return getResponse(ints, intents)

def no_answer(question: str, username: str):
    url = "http://127.0.0.1:8000/api/unanswered"
    data = {"question": question, "sender": username}
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()  # Raises an error for bad responses
        print("Question sent successfully:", response.status_code)
    except requests.exceptions.RequestException as e:
        print("Error in sending question:", e)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message_type: str = update.message.chat.type
    text: str = update.message.text
    username: str = update.message.from_user.username
    
    print(f'User({update.message.chat.id}) in {message_type}: "{text}"')
    
    if message_type == 'group':
        if BOT_USERNAME in text:
            new_text: str = text.replace(BOT_USERNAME, '').strip()
            response = handle_response(new_text, username)
        else:
            return
    else:
        response = handle_response(text, username)
    
    print('Bot:', response)
    await update.message.reply_text(response)
    
    # Send chat history to the API
    try:
        ints = predict_class(text, model)
        confidence = float(ints[0]['probability']) if ints else 0.0
        chat_history = {
            "sender": username,
            "message": text,
            "response": response,
            "confidence": confidence
        }
        url = "http://127.0.0.1:8000/api/chat_history"
        api_response = requests.post(url, json=chat_history)
        api_response.raise_for_status()
        print("Chat history sent successfully:", api_response.status_code)
    except requests.exceptions.RequestException as e:
        print("Error in sending chat history:", e)

async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update is not None:
        print(f'Update {update} caused error {context.error}')
        await update.message.reply_text('Maaf, Silahkan bertanya kembali')
    else:
        print(f'Error occurred: {context.error}')

if __name__ == '__main__':
    print('Starting bot...')
    load_data()  # Initial load
    app = Application.builder().token(TOKEN).build()
    
    # Commands
    app.add_handler(CommandHandler('start', start_command))
    app.add_handler(CommandHandler('help', help_command))
    app.add_handler(CommandHandler('custom', custom_command))
    # app.add_handler(CommandHandler('link', link_command))
    
    # Messages
    app.add_handler(MessageHandler(filters.TEXT, handle_message))
    
    # Errors
    app.add_error_handler(error)
    
    # Polls the bot
    print('Polling...')
    app.run_polling(poll_interval=3)

