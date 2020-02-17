
from flask import Flask, render_template, request
import flask
import time
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
import wikipedia
from sklearn.model_selection import train_test_split
import warnings
from itertools import groupby
import pyttsx3

warnings.filterwarnings("ignore", category=DeprecationWarning)
from rake_nltk import Rake
import speech_recognition as s
from googleplaces import GooglePlaces, types, lang
import requests
import json
import geocoder
from geopy.geocoders import Nominatim
import urllib
import sys
from bs4 import BeautifulSoup

API_KEY = 'AIzaSyBwIxGF4R3SPITPcW46D1nU2qiTMibfkIs'
engine = pyttsx3.init()
training = pd.read_csv('Training.csv')
testing = pd.read_csv('Testing.csv')
cols = training.columns
cols = cols[:-1]
x = training[cols]
y = training['prognosis']
y1 = y
reduced_data = training.groupby(training['prognosis']).max()
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx = testing[cols]
testy = testing['prognosis']
testy = le.transform(testy)
clf1 = DecisionTreeClassifier()
clf = clf1.fit(x_train, y_train)
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

inp11 = ""
ip10 = ""
inp10 = ""
a = 0
b = 0
c = 0
d = 0
e = 0
f = 0
x = 0
z = 0
ip10 = ""
val = ""
val1 = ""
m = ""
msg1 = ""
msg = ""
symptoms_present = []
symptoms_given = []
n10 = 0
depth = 1
flag = 0
f1 = 0
f2 = 0
global tree_
global feature_name
app = Flask(__name__)


def googler(query):
    query = query.replace(' ', '+')
    URL = f"https://google.com/search?q={query}"
    USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:65.0) Gecko/20100101 Firefox/65.0"
    headers = {"user-agent": USER_AGENT}
    resp = requests.get(URL, headers=headers)
    if resp.status_code == 200:
        soup = BeautifulSoup(resp.content, "html.parser")
    results = []
    for g in soup.find_all('div', class_='r'):
        anchors = g.find_all('a')
        if anchors:
            link = anchors[0]['href']
            title = g.find('h3').text
            item = {
                "title": title,
                "link": link
            }
            results.append(item)
    return (str(results) + "You can click on the links here")


def places(place):
    try:
        google_places = GooglePlaces(API_KEY)
        geolocator = Nominatim()
        addr = place
        loc = geolocator.geocode(addr, timeout=50000)
        lat1 = loc.latitude
        lon1 = loc.longitude

        query_result = google_places.nearby_search(
            lat_lng={'lat': lat1, 'lng': lon1},
            types=[types.TYPE_HOSPITAL])
        # if query_result.has_attributions:
        # print(query_result.html_attributions)
        return ("Hospitals in your city are:" +query_result.html_attributions+ query_result.places)
    except:
        sen31 = "Sorry,I am unable to process your request"
        return (sen31)


@app.route('/')
def dyn_page():
    return render_template('home.html')


@app.route('/get')
def get_bot_response():
    global msg
    msg = request.args.get('msg')
    global m
    global val1
    global d
    global f2
    f2 = 0
    if d < 3 or "i" not in ip10:
        d += 1
        if msg == "tell":
            print("jnnfs")
            val1 = msg
            m = ""
            user_input = tree_to_code(m, clf, cols)
        elif (msg == "text"):
            print("bjs")
            val1 = msg
            m = ""
            user_input = tree_to_code(m, clf, cols)

        elif ((msg == "yes" or msg == "no") and (f2 == 1)):
            if (msg == "no"):
                print("hello")
                user_input = rediagnosis()
            else:
                print("hii")
                msg = request.args.get('msg')
                user_input = sym()
        elif (msg == "yes" and f2 == 0):
            print("11")
            f2 += 1
            user_input = tree_to_code(msg, clf, cols)
        else:
            print("hhiwa")
            user_input = tree_to_code(msg, clf, cols)
    else:
        print("oyee")
        user_input = sym()
    print(user_input)
    return flask.jsonify(user_input)


def print_disease(node):
    # print(node)
    node = node[0]
    # print(len(node))
    val = node.nonzero()
    # print(val)
    disease = le.inverse_transform(val[0])
    return disease


def enter():
    global e
    global val1
    global msg1
    if e == 0:
        if val1 == "tell":
            e += 1
            sen10 = "Tell me all the symptoms that you have"
            return (sen10)
        else:
            e += 1
            sen10 = 'Enter all the symptoms you have'
            return (sen10)

    df = pd.DataFrame(training, columns=training.columns.values)
    col_list = df.columns.tolist()
    r = Rake()
    a = r.extract_keywords_from_text(val)
    b = r.get_ranked_phrases()
    count = 0
    max_count = 0
    found = 0
    symptoms_given = []
    if (any(x in b for x in col_list)):
        for ind in df.index:
            for i in b:
                if (i in col_list and df.at[ind, i] == 1):
                    count += 1

            if (count > max_count):
                max_count = count
                found = ind
        present_disease = df.at[found, 'prognosis']
        msg1 = "I think you have " + " " + present_disease

        return (b, msg1)
    else:
        sen21 = "Sorry,I am not able to predict your disease.Take care,bye"
        return sen21
        sys.exit()


def tree_to_code(ip, tree, feature_names):
    global a
    global msg1
    if a == 0:
        if val1 == "tell":
            sen11 = "Do you want to tell symptoms or should I ask?"
            a += 1
            return (sen11)
        else:
            sen11 = "Do you want to type symptoms or should I ask?"
            a += 1
            return (sen11)
    else:
        global symptoms_present
        global symptoms_given
        global c
        global ip10
        global val
        global f1
        global tree_
        global feature_name
        if c < 2:
            ip10 = ip.lower()
            if "i" in ip10:
                if c == 0:
                    c += 1
                    present_disease = enter()
                    return (present_disease)
                else:
                    c += 1
                    val = ip
                    a = enter()
                    symptoms_present = a[0]
                    msg1 = a[1]
            else:
                if (f1 == 0):
                    f1 += 1
                    c += 1
                    sen3 = "Please reply Yes or No for the following symptoms"
                    tree_ = tree.tree_
                    # print(tree_)
                    feature_name = [
                        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
                        for i in tree_.feature

                    ]
                    return (sen3)
                else:
                    def recurse(node, d):
                        global symptoms_present
                        global symptoms_given
                        global tree_
                        global feature_name
                        global flag
                        global n10
                        global depth
                        global msg
                        global msg11
                        indent = " " * (d)
                        if tree_.feature[node] != _tree.TREE_UNDEFINED:
                            name = feature_name[node]
                            name = name.replace("_", " ")
                            threshold = tree_.threshold[node]
                            if (flag == 0):
                                indent = " " * (d)
                                flag = 1
                                return (name + " ?")
                            else:
                                flag = 0
                                ans = msg.lower()
                                if ans == 'yes':
                                    val = 1
                                else:
                                    val = 0
                                if val <= threshold:
                                    n10 = tree_.children_left[node]
                                    depth = d + 1
                                    d = d + 1
                                    return (recurse(n10, d))
                                else:
                                    symptoms_present.append(name)
                                    n10 = tree_.children_right[node]
                                    depth = d + 1
                                    d = d + 1
                                    return (recurse(n10, d))
                        else:
                            present_disease = print_disease(tree_.value[node])
                            for i in present_disease:
                                sen = "I think you have " + i + "\n"
                            a = " "
                            red_cols = reduced_data.columns
                            symptoms_given.append(red_cols[reduced_data.loc[present_disease].values[0].nonzero()])
                            for i in symptoms_given:
                                for j in i:
                                    a = a + "\n," + j
                            msg11 = sen + str(a)
                            return (sym())
                d = depth
                an = (recurse(n10, d))
                return an
            msg1 = sym()
            return msg1


def sym():
    global msg1
    global msg11
    global msg
    global inp10
    global inp11
    global f
    global z
    ip1 = ""
    if "i" not in ip10:
        if (z == 0):
            z += 1
            return (msg11 + "\n Do you have any of these symptoms?")
        else:
            inp10 = msg
            inp11 = inp10.lower()
    else:
        inp11 = "yes"
    if (inp11 == "yes" or "i" not in ip10):
        if (f == 0):
            f += 1
            sen5 = str(msg1) + " " + "Do you want to know more about your disease? Enter yes or no"
            return sen5
        else:
            if f == 1:
                f += 1
                ip1 = msg
                if (f == 2):
                    if (ip1 == "yes"):
                        sen4 = "You can read more about your disease here..."
                        for i in symptoms_present:
                            try:
                                i = i.lower()
                                sen6 = sen4 + "\n" + (wikipedia.summary(i, sentences=20))
                            except:
                                return (
                                            "Sorry, couldn't get more information" + "\n Do you want to know which are the hospitals in your city?")
                        return sen6 + "\n Do you want to know which are the hospitals in your city?"
                    else:
                        return "Do you want to know which are the hospitals in your city?"
            else:
                if (f == 2):

                    ip2 = msg
                    if (ip2 == "yes"):
                        return "Enter your city,country"
                    elif (ip2 == "no"):
                        f += 1
                        return "Do you want the link for home remedies for your disease?"
                    else:
                        f += 1
                        s = places(msg)
                        sen7 = s + "\n Do you want the link for home remedies for your disease?"
                        return sen7
                elif (f == 3):
                    f += 1
                    ip3 = msg
                    if (ip3 == "yes"):
                        for i in symptoms_present:
                            i = i.lower()
                            sen8 = googler(i + "home remedies")
                    else:
                        sen8 = ""
                    return sen8 + "\n Do you want to know few medicines for your disease?"
                elif (f == 4):
                    f += 1
                    ip4 = msg
                    if (ip4 == "yes"):
                        for i in symptoms_present:
                            i = i.lower()
                        return (googler("Medicines for " + i) + "\n Stay Healthy, Take care ,Bye")
                    else:
                        return ("Stay Healthy,Take care,bye")
                else:
                    pass


def rediagnosis():
    global y
    sen30 = "You have to be rediagnosed"
    return sen30
    recurse(n10, d)
    sym()


if __name__ == '__main__':
    app.run(debug=True)