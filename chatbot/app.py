from flask import Flask,render_template,request
import flask
import time
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier,_tree
import numpy as np
import wikipedia
from sklearn.model_selection import train_test_split
import warnings
from itertools import groupby
import pyttsx3
warnings.filterwarnings("ignore", category=DeprecationWarning)
from rake_nltk import Rake
import speech_recognition as s
from googleplaces import GooglePlaces,types,lang
import requests
import json
import geocoder
from geopy.geocoders import Nominatim
import urllib
import sys
from bs4 import BeautifulSoup
app=Flask(__name__)

@app.route('/')
def dyn_page():
    return render_template('home.html')

global val1  
  
@app.route('/get')
def get_bot_response():
    val1=request.args.get('msg')
    user_input=tree_to_code(val1,clf,cols)
    return flask.jsonify(user_input)

def sample(val1):
    if val1=="tell":
        s="ananya"
        speak(s)
        s="ananya"
        print(s)
        return (s)
        r=input()
    else:
        
        s="anurag"
        speak(s)
        print(s)
        return s	
    
API_KEY='AIzaSyCpFe5FfiTuTZHiOEy8zZXcYMojIyeNJfQ'
engine=pyttsx3.init()
training=pd.read_csv('Training.csv')
testing=pd.read_csv('Testing.csv')
cols     = training.columns
cols     = cols[:-1]
x        = training[cols]
y        = training['prognosis']
y1       = y
reduced_data = training.groupby(training['prognosis']).max()
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx    = testing[cols]
testy    = testing['prognosis']
testy    = le.transform(testy)
clf1  = DecisionTreeClassifier()
clf = clf1.fit(x_train,y_train)
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols
def speech():
    sr=s.Recognizer()
    print("I am listening to your words...")
    with s.Microphone() as m:
        audio=sr.listen(m)
        val=sr.recognize_google(audio,language='eng-in')
        print(val)
    return val


def speak(sen):
    newRate=180
    engine.setProperty('rate',newRate)
    engine.say(sen)
    engine.runAndWait()

def enter(s):
    df = pd.DataFrame(training, columns=training.columns.values)
    col_list = df.columns.tolist()
    val1=s
    if val1=="tell":
        sen10="Tell me all the symptoms that you have"
        speak(sen10)
        print(sen10) 
        val=speech()
    else:
        sen10 = 'Enter all the symptoms you have'
        speak(sen10)
        return (sen10)
        val=input()

    r = Rake()
    a = r.extract_keywords_from_text(val)
    b = r.get_ranked_phrases()
    count = 0
    max_count = 0
    found = 0
    if (any(x in b for x in col_list)):
        for ind in df.index:
            for i in b:
                if (i in col_list and df.at[ind, i] == 1):
                    count += 1

            if (count > max_count):
                max_count = count
                found = ind
        present_disease = df.at[found, 'prognosis']
        speak("I think you have " + present_disease)
        print("I think you have " + present_disease)
        symptoms_present.append(present_disease)
        return present_disease
    else:
        sen21="Sorry,I am not able to predict your disease"
        speak(sen21)
        return(sen21)
        speak("Take care,bye")
        sys.exit()

def tree_to_code(val1,tree, feature_names):
    if val1=="tell":
        sen11 = "Do you want to tell symptoms or should I ask?"
        #speak(sen11)
        print(sen11)
        return sen11
    else:
        sen11 = "Do you want to type symptoms or should I ask?"
        #speak(sen11)
        return (sen11)
        ip10 = input()
    ip10 = ip10.lower()
    symptoms_present = []
    if "i" in ip10:
        present_disease=enter(symptoms_present)
    else:
        sen3 = "Please reply Yes or No for the following symptoms"
        #speak(sen3)
        print(sen3)
        tree_ = tree.tree_
        #print(tree_)
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
 

if __name__ == '__main__':
    app.run(debug=True)