import random
from keras.optimizers import SGD
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping # Import EarlyStopping
import numpy as np
import pickle
import json
import requests
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import sys
import nltk
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split # Import train_test_split

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# Define the API endpoint
url = 'http://127.0.0.1:8000/api/faq'

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

# Shuffle our features
random.shuffle(training)
all_data_np = np.array(training, dtype=object)

# Create train, validation, and (optionally) test lists.
# X - patterns, Y - intents
all_x = np.array(list(all_data_np[:, 0]), dtype=np.float32)
all_y = np.array(list(all_data_np[:, 1]), dtype=np.float32)

# Split data into training and validation sets (e.g., 80% train, 20% validation)
train_x, val_x, train_y, val_y = train_test_split(all_x, all_y, test_size=0.2, random_state=42) # random_state for reproducibility

print("Training data shape:", train_x.shape, train_y.shape)
print("Validation data shape:", val_x.shape, val_y.shape)
print("Training and validation data created")

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

# Define EarlyStopping callback
# It will monitor 'val_loss'. Training will stop if 'val_loss' doesn't improve for 'patience' epochs.
# 'restore_best_weights=True' will ensure that the model weights are restored to those of the epoch with the best 'val_loss'.
early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True) # Increased patience a bit

# Fitting and saving the model
# Pass validation_data and the early_stopping callback
hist = model.fit(train_x, train_y, epochs=350, batch_size=5, verbose=1,
validation_data=(val_x, val_y), # Add validation data
callbacks=[early_stopping])      # Add EarlyStopping callback

model.save('model.h5') # Save the model (it will have the best weights if restore_best_weights=True)
print("Model created and saved with best weights based on validation loss.")

# --- Evaluation ---
# Now, you should ideally have a separate TEST set that the model has never seen
# For demonstration, we'll evaluate on the validation set (which is better than training set)
# but a true test set is better for final unbiased evaluation.

val_y_pred_probs = model.predict(val_x)
val_y_pred = np.argmax(val_y_pred_probs, axis=1)
val_y_true = np.argmax(val_y, axis=1)

accuracy = accuracy_score(val_y_true, val_y_pred)
precision = precision_score(val_y_true, val_y_pred, average='weighted', zero_division=0) # Added zero_division
recall = recall_score(val_y_true, val_y_pred, average='weighted', zero_division=0) # Added zero_division

print(f"Validation Accuracy: {accuracy}")
print(f"Validation Precision: {precision}")
print(f"Validation Recall: {recall}")

# Plotting accuracy and loss
accuracy_hist = hist.history['accuracy']
val_accuracy_hist = hist.history['val_accuracy'] # Get validation accuracy
loss_hist = hist.history['loss']
val_loss_hist = hist.history['val_loss'] # Get validation loss

# Determine the number of epochs that were actually run
# If early stopping occurred, this will be less than 350
actual_epochs = range(1, len(accuracy_hist) + 1)

# Plotting accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(actual_epochs, accuracy_hist, 'b-', label='Training accuracy')
plt.plot(actual_epochs, val_accuracy_hist, 'g-', label='Validation accuracy') # Plot validation accuracy
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plotting loss
plt.subplot(1, 2, 2)
plt.plot(actual_epochs, loss_hist, 'b-', label='Training loss')
plt.plot(actual_epochs, val_loss_hist, 'g-', label='Validation loss') # Plot validation loss
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# If you didn't use restore_best_weights=True with EarlyStopping,
# you could manually find the epoch with the lowest validation loss:
if not early_stopping.restore_best_weights: # Or if you just want to know the epoch number
    best_epoch_val_loss = np.argmin(hist.history['val_loss']) + 1 # +1 because epochs are 1-indexed
    best_val_loss = np.min(hist.history['val_loss'])
    print(f"Best epoch based on lowest validation loss: {best_epoch_val_loss} (Loss: {best_val_loss:.4f})")

    best_epoch_val_acc = np.argmax(hist.history['val_accuracy']) + 1
    best_val_acc = np.max(hist.history['val_accuracy'])
    print(f"Best epoch based on highest validation accuracy: {best_epoch_val_acc} (Accuracy: {best_val_acc:.4f})")
    # You would then typically retrain your model for 'best_epoch_val_loss' number of epochs on the *entire* training data (train_x + val_x)
    # or load a checkpoint if you saved them.
