import requests
import json
import random
import numpy as np
import pickle
import nltk
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

# Load the model and intents
model = load_model('model.h5')
intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))

# Print the vocabulary size during inference
print(f"Vocabulary size during inference: {len(words)}")

TOKEN: Final = '7550700421:AAGkbxFxXU-wqtmzeyzVPJsj1vVIROODo_s'
BOT_USERNAME: Final = '@skabum_lawang_chatbot'

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

async def link_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('https://www.youtube.com/watch?v=vZtm1wuA2yc')

# Responses
def handle_response(text: str) -> str:
    ints = predict_class(text, model)
    if not ints or len(ints) == 0:
        return "Maaf saya tidak tau"
    
    top_intent = ints[0]
    tag = top_intent['intent']
    confidence = float(top_intent['probability'])
    
    if confidence < 0.25:  # Adjust the threshold as needed
        return "Maaf saya tidak tau"
    
    if tag=='noanswer':
       no_answer(text)
    
    return getResponse(ints, intents)

def no_answer(question: str):
    url = "http://chatbot.test/api/unanswered"
    data = {"question": question}
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()  # Raises an error for bad responses
        print("Question sent successfully:", response.status_code)
    except requests.exceptions.RequestException as e:
        print("Error in sending question:", e)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message_type: str = update.message.chat.type
    text: str = update.message.text
    
    print(f'User({update.message.chat.id}) in {message_type}: "{text}"')
    
    if message_type == 'group':
        if BOT_USERNAME in text:
            new_text: str = text.replace(BOT_USERNAME, '').strip()
            response = handle_response(new_text)
        else:
            return
    else:
        response = handle_response(text)
        
    print('Bot:', response)
    await update.message.reply_text(response)

async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update is not None:
        print(f'Update {update} caused error {context.error}')
        await update.message.reply_text('Maaf, Silahkan bertanya kembali')
    else:
        print(f'Error occurred: {context.error}')

if __name__ == '__main__':
    print('Starting bot...')
    app = Application.builder().token(TOKEN).build()
    
    # Commands
    app.add_handler(CommandHandler('start', start_command))
    app.add_handler(CommandHandler('help', help_command))
    app.add_handler(CommandHandler('custom', custom_command))
    app.add_handler(CommandHandler('link', link_command))
    
    # Messages
    app.add_handler(MessageHandler(filters.TEXT, handle_message))
    
    # Errors
    app.add_error_handler(error)
    
    # Polls the bot
    print('Polling...')
    app.run_polling(poll_interval=3)
