import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier,_tree
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import warnings
from itertools import groupby
warnings.filterwarnings("ignore", category=DeprecationWarning)
from rake_nltk import Rake
import collections
training = pd.read_csv('Training.csv')
testing  = pd.read_csv('Testing.csv')
cols     = training.columns
cols     = cols[:-1]
x        = training[cols]
y        = training['prognosis']
y1       = y
df=pd.DataFrame(training, columns=training.columns.values)
col_list=df.columns.tolist()
val=input('Enter all the symptoms you have')
r=Rake()
a=r.extract_keywords_from_text(val)
b=r.get_ranked_phrases()
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

def predict(b,col_list):
    count = 0
    max_count = 0
    found = 0
    if(any(x in b for x in col_list)):
        for ind in df.index:
            for i in b:
                if(i in col_list and df.at[ind,i]==1):
                    count+=1

            if(count>max_count):
                max_count=count
                found=ind

        print("You may have "+df.at[found,'prognosis'])

    else:
        print("Unpredictable")


predict(b,col_list);




