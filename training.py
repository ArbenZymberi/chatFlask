import tensorflow as tf
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import json
import nltk 
import pickle

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

lemmatizer = WordNetLemmatizer()
#intents = json.loads(open('data.json', 'r')).read()
# with open('data.json', 'r') as f:
#     intents = json.load(f)

intents = json.load(open('data.json', 'r'))

words = []
classes = []
documents = []
ignore_letters = [',','?','!','.','/','\'']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        #Stores the token in the array words
        words.extend(word_list)
        #This will store the token in the documents array together with the "tag" value from data.json
        documents.append((word_list, intent['tag']))
        #This checks the tag has been added to the array classes
        if intent['tag'] not in classes:
            #If it has not been added, it will add the tag to it
            classes.append(intent['tag'])

#At this stage we have knowladge of the avilable data

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]

#Optional code to add to see how the data is being stored
#print(classes)
#print(documents)
#print(words)

words = sorted(set(words))
classes = sorted(set(classes))

#Pickle the data
#explain what pickling does, and why we are doing it for this file to be added later
#Reduce size of file for transfer
#Changing file type to transfer to then change to original at destination
#wb stands for write binary
pickle.dump(words, open('words.pk1', 'wb'))
pickle.dump(classes, open('classes.pk1','wb'))

#To run this code in terminal use       py training.py     in terminal
#This should now create two files in the directory
#If it does not, check spellings, and antivirus blocking the running of the file

#######################################################################################################

training = []
output_empty = [0] * len(classes)

#Turning the data into a bag of words
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])
    #print(bag)

#making the training data biased
random.shuffle(training)
training = np.array(training)

#Spliting the data into lists of 0s and 1s
train_x = list(training[:,0])
train_y = list(training[:,1])

model = tf.keras.Sequential()
#first layer of our neural network, 128 neurons
#If you increase the data, you can increase the neurons for capacity 
model.add(tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),),activation='relu'))
#Using a dropout
model.add(tf.keras.layers.Dropout(0.5))
#Second layer of neurons
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
#Third layer of neurons which is dynamic
model.add(tf.keras.layers.Dense(len(train_y[0]), activation='softmax'))


#Using an optimiser called SGD Stochastic Gradient Descent algorithm
sgd = tf.keras.optimizers.legacy.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

#How much training it needs to do
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#How this model will be fit and how many epoch it will go through
history = model.fit(np.array(train_x), np.array(train_y), epochs=100, batch_size=32, verbose=1)

#Saving the model with accuracy data to see how accurate it is
model.save('chatbot_model.model', history)

#When your run this code and it gives you an issue, you want have to downgrade the installation file 
#of numpy to 1.23.5
#   pip install --upgrade numpy==1.23.5
