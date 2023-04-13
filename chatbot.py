import json
import random
import pickle
import tensorflow as tf
import numpy as np
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
intents = json.load(open('data.json', 'r'))

words = pickle.load(open('words.pk1', 'rb'))
classes = pickle.load(open('classes.pk1', 'rb'))
model = tf.keras.models.load_model("chatbot_model.model")

def clean_up_sentence(sentence):
    sentence_words = word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words] 
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for x in sentence_words:
        for i, word in enumerate(words):
            if word == x:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.20
    results = [[i,r] for i,r, in enumerate(res) if r > ERROR_THRESHOLD]
    result_list = []
    for r in results:
        result_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return result_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print ("Chatbor is running, say something...")
while True:
    message = input("")
    message = message.lower()
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)