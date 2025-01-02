import time
import random
from keras.optimizers import SGD
from keras.layers import Dense, Dropout
from keras.models import Sequential
import numpy as np
import pickle
import json
import requests
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import sys
import nltk

# Start the timer
start_time = time.time()

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# Define the API endpoint
url = 'http://chatbot.test/api/faq'

# Send a GET request to the API
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    api_response = response.json()
    # Extract only the 'intents' data
    intents = api_response['data']

    # Save the intents data to a JSON file
    with open('data.json', 'w') as json_file:
        json.dump(intents, json_file, indent=2)

    print("Intents data fetched and saved successfully!")
else:
    print(f"Failed to fetch data. Status code: {response.status_code}")
    sys.exit()

# Initialize the stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('data.json').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Add documents in the corpus
        documents.append((w, intent['tag']))

        # Add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# Print the vocabulary size
print(f"Vocabulary size during training: {len(words)}")

# Sort classes
classes = sorted(list(set(classes)))

# Save the output to a text file
output_file = 'output.txt'
with open(output_file, 'w') as f:
    f.write(f"{len(documents)} documents\n")
    f.write(f"{len(classes)} classes: {classes}\n")
    f.write(f"{len(words)} unique stemmed words: {words}\n")

pickle.dump(words, open('texts.pkl', 'wb'))
pickle.dump(classes, open('labels.pkl', 'wb'))

print(f"Output saved to {output_file}")

# Create our training data
training = []
# Create an empty array for our output
output_empty = [0] * len(classes)
# Training set, bag of words for each sentence
for doc in documents:
    # Initialize our bag of words
    bag = [0] * len(words)
    # List of tokenized words for the pattern
    pattern_words = doc[0]
    # Stem each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # Create our bag of words array with 1, if word match found in current pattern
    for w in pattern_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1

    # Output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# Shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training, dtype=object)
# Create train and test lists. X - patterns, Y - intents
train_x = np.array(list(training[:, 0]), dtype=np.float32)
train_y = np.array(list(training[:, 1]), dtype=np.float32)
print("Training data created")

# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Fitting and saving the model
hist = model.fit(train_x, train_y, epochs=1000, batch_size=5, verbose=1)
model.save('model.h5')

print("Model created")

# End the timer
end_time = time.time()

# Calculate the duration
execution_time = end_time - start_time

# Print the execution time
print(f'Execution time: {execution_time} seconds')
