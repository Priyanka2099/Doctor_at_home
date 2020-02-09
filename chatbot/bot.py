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
from bs4 import BeautifulSoup
API_KEY='AIzaSyCpFe5FfiTuTZHiOEy8zZXcYMojIyeNJfQ'
engine=pyttsx3.init()
def speak(sen):
    newRate=180
    engine.setProperty('rate',newRate)
    engine.say(sen)
    engine.runAndWait()
def googler(query):
    query=query.replace(' ','+')
    URL=f"https://google.com/search?q={query}"
    USER_AGENT= "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:65.0) Gecko/20100101 Firefox/65.0"
    headers= {"user-agent" :USER_AGENT}
    resp=requests.get(URL,headers=headers)
    if resp.status_code==200:
        soup=BeautifulSoup(resp.content,"html.parser")
    results=[]
    for g in soup.find_all('div',class_='r'):
        anchors=g.find_all('a')
        if anchors:
            link=anchors[0]['href']
            title=g.find('h3').text
            item= {
                "title": title,
                "link": link
            }
            results.append(item)
    print(results)
    speak("You can click on the links here")

def places():
    google_places=GooglePlaces(API_KEY)
    geolocator=Nominatim()
    sen0="Enter your city,country"
    speak(sen0)
    print(sen0)
    addr=input()
    loc=geolocator.geocode(addr,timeout=50000)
    lat1=loc.latitude
    lon1=loc.longitude

    query_result=google_places.nearby_search(
        lat_lng={'lat': lat1 , 'lng': lon1},
        types=[types.TYPE_HOSPITAL])
    speak("Hospitals in your city are:")
    if query_result.has_attributions:
        print(query_result.html_attributions)
    for place in query_result.places:
        print(place.name)

training=pd.read_csv('Training.csv')
testing=pd.read_csv('Testing.csv')
cols     = training.columns
cols     = cols[:-1]
x        = training[cols]
y        = training['prognosis']
y1       = y
reduced_data = training.groupby(training['prognosis']).max()

#mapping strings to numbers
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

sen1="Hi, I am a chatbot,I can predict disease based on symptoms"
speak(sen1)
print(sen1)
sen2="You can either type or tell, enter text or tell based on your preference"
speak(sen2)
print(sen2)
val1=input()

def speech():
    sr=s.Recognizer()
    print("I am listening to your words...")
    with s.Microphone() as m:
        audio=sr.listen(m)
        val=sr.recognize_google(audio,language='eng-in')
        print(val)
    return val
def print_disease(node):
    #print(node)
    node = node[0]
    #print(len(node))
    val = node.nonzero()
    #print(val)
    disease = le.inverse_transform(val[0])
    return disease
def enter(symptoms_present):

    df = pd.DataFrame(training, columns=training.columns.values)
    col_list = df.columns.tolist()
    sen10='Enter all the symptoms you have'
    speak(sen10)
    print(sen10)
    val = input()
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
        print("Sorry,I am not able to predict your disease")
def tree_to_code(tree, feature_names):
    sen11="Do you want to enter symptoms or should I ask?"
    speak(sen11)
    print(sen11)
    ip10=input()
    ip10=ip10.lower()
    symptoms_present = []
    if "i" in ip10:
        present_disease=enter(symptoms_present)
    else:
        sen3 = "Please reply Yes or No for the following symptoms"
        speak(sen3)
        print(sen3)
        tree_ = tree.tree_
        #print(tree_)
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        def recurse(node, depth):
            indent = "  " * depth
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                name=name.replace("_"," ")
                threshold = tree_.threshold[node]
                speak(name)
                print(name + " ?")
                if(val1=="tell"):
                    ans=speech()
                elif(val1=="text"):
                    ans = input()
                ans = ans.lower()
                if ans == 'yes':
                    val = 1
                else:
                    val = 0
                if  val <= threshold:
                    recurse(tree_.children_left[node], depth + 1)
                else:
                    symptoms_present.append(name)
                    recurse(tree_.children_right[node], depth + 1)
            else:
                present_disease = print_disease(tree_.value[node])
                for i in present_disease:
                    speak("I think you have"+i)
                    print( "I think you have " + i )

                red_cols = reduced_data.columns
                symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
                for i in symptoms_present:
                    i=i.replace("_"," ")
                    speak("symptoms that you have is " + i)
                    print("symptoms that you have is " + i)
                    print("Other general symptoms for this disease are ")
                for i in symptoms_given:
                    print(i)
                confidence_level = (1.0 * len(symptoms_present)) / len(symptoms_given)
                print("confidence level is " + str(confidence_level))
    sen5="Do you want to know more about your disease? Enter yes or no"
    speak(sen5)
    print(sen5)
    ip1=input()
    if (ip1 == "yes"):
        sen4 = "You can read more about your disease here..."
        speak(sen4)
        print(sen4)
        for i in symptoms_present:
            try:
                i = i.lower()
                print(wikipedia.summary(i, sentences=20))
            except:
                print("Sorry, couldn't get more information")
    sen6 = "Do you want to know which are the hospitals in your city?"
    speak(sen6)
    print(sen6)
    ip2 = input()
    if (ip2 == "yes"):
        places()
    sen7 = "Do you want the link for home remedies for your diseasse?"
    speak(sen7)
    print(sen7)
    ip3 = input()
    if (ip3 == "yes"):
        for i in symptoms_present:
            i = i.lower()
            googler(i + "home remedies")
    sen8 = "Do you want to know few medicines for you disease?"
    speak(sen8)
    print(sen8)
    ip4 = input()
    if (ip4 == "yes"):
        for i in symptoms_present:
            i = i.lower()
            googler("Medicines for " + i)
    else:
        print("Take care,bye")


    recurse(0, 1)

tree_to_code(clf, cols)
