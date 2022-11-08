# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


#This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
import string
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import joblib
from keras.models import load_model
import re

class ActionHelloWorld(Action):

    def name(self) -> Text:
        return "action_hello_world"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        #Getting the latest user message.
        for event in tracker.events:
            if event.get("event") == "user":
                latestMessage = event.get("text")

        dispatcher.utter_message(text=latestMessage)
        return [SlotSet("classification_result", "true")]

class FakeNewsClassifier(Action):

    def __init__(self):
        self.vectorizer = CountVectorizer(max_features=10000)
        self.lemma = WordNetLemmatizer()
        #self.downloadNltkDependencies()  
        self.createVectorizer()

    def name(self) -> Text:
        return "action_classifier"

    def downloadNltkDependencies():
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('omw-1.4')

    def createVectorizer(self):
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        df_preprocessing = pd.read_csv("actions\\final_database_pre_processada.csv")
        df_preprocessing.drop(df_preprocessing.columns[0], axis=1, inplace=True)
        df_array = df_preprocessing["0"].values.tolist()
        BagOfWOrds = self.vectorizer.fit_transform(df_array)
        

    def preprocessingMessage(self, text):
        pontuacoes = string.punctuation
        stop_words = stopwords.words("portuguese")
        text = text = text.lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub("[^a-zA-Z0-9]"," ",text)
        text = nltk.word_tokenize(text.lower())
        text = [self.lemma.lemmatize(word) for word in text]
        text = [word for word in text if word not in stop_words and word not in pontuacoes]
        text = " ".join(text)
        return text

    def encodeMessage(self, text):
        preprocessed_text = self.preprocessingMessage(text)
        encoded_text = self.vectorizer.transform([preprocessed_text]).toarray()
        return encoded_text

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        #Getting the latest user message.
        for event in tracker.events:
            if event.get("event") == "user":
                latestMessage = event.get("text")

        #Encoding message.
        encoded_value = self.encodeMessage(latestMessage)

        loaded_model = load_model('actions\my_CNN_model.h5')

        predicted_value = loaded_model.predict(encoded_value)
        
        value = str(predicted_value[0][0]*100)
        
        #Check if dispatcher is necessary.
        dispatcher.utter_message(text=latestMessage)

        #Insert classifier logic.
        return [SlotSet("classification_result", value)]

