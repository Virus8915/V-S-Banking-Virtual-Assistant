import streamlit as st
import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split as tts
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder as LE
from sklearn.metrics.pairwise import cosine_similarity

import nltk
nltk.download('punkt')
from nltk.corpus import stopwords

import nltk
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def cleanup(sentence):
    word_tok = nltk.word_tokenize(sentence)
    stemmed_words = [w for w in word_tok if not w in stop_words]
    return ' '.join(stemmed_words)

le = LE()
tfv = TfidfVectorizer(min_df=1, stop_words='english')

df3 = pd.read_csv("BankFAQs.csv")
questions = df3['Question'].values

X = []
for question in questions:
    X.append(cleanup(question))

tfv.fit(X)
le.fit(df3['Class'])
X = tfv.transform(X)
y = le.transform(df3['Class'])

trainx, testx, trainy, testy = tts(X, y, test_size=.3, random_state=42)
model = SVC(kernel='linear')
model.fit(trainx, trainy)

class_=le.inverse_transform(model.predict(X))

def get_max5(arr):
    ixarr = []
    for ix, el in enumerate(arr):
        ixarr.append((el, ix))

    ixarr.sort()

    ixs = []
    for i in ixarr[-5:]:
        ixs.append(i[1])

    return ixs[::-1]

def get_response(usrText):
    while True:
        if usrText.lower() == "bye":
            return "Bye"

        GREETING_INPUTS = ["hello","hey", "hi"]
        a = [x.lower() for x in GREETING_INPUTS]

        sd=["Thanks"]
        d = [x.lower() for x in sd]

        am=["ok"]
        c = [x.lower() for x in am]

        t_usr = tfv.transform([cleanup(usrText.strip().lower())])
        class_ = le.inverse_transform(model.predict(t_usr))

        questionset = df3[df3['Class'].values == class_]

        cos_sims = []
        for question in questionset['Question']:
            sims = cosine_similarity(tfv.transform([question]), t_usr)
            cos_sims.append(sims)

        ind = cos_sims.index(max(cos_sims))
        b = [questionset.index[ind]]

        if usrText.lower() in a:
            return "Hi, I'm your BankingBuddy! How can I help you?"

        if usrText.lower() in c:
            return "Ok...Alright!"

        if usrText.lower() in d:
            return "My pleasure!"

        if max(cos_sims) > 0.0:
            return df3['Answer'][questionset.index[ind]] + " "

        elif max(cos_sims) == 0.0:
            return "Sorry, Can you please rephrase your question?"
import streamlit as st
import os

import streamlit.components.v1 as com


com.html(


<html lang="en">
    <head>
        <meta charset="UTF-8">
        <style>
            body{
                font-family: Tahoma, Geneva, sans-serif;
            }
            
            .box{
                width: 70vw;
                margin: auto;
            }
            
            .chat{
                width: 60vw;
                padding: 2em;
                border-radius: .5em;
                font-size: 1.2em;
            }
            
            .user{
                text-align: left;
                background: #836fe6; 
                color: rgb(247, 247, 247); 
            }
            
            .bot{
                margin-left: 5vw;
                text-align: right;
                background: #f55182; 
                color: white;
            }
        </style>

    </head>
    <body>
        <h1 align="center">
            V&S Banking Virtual Assistant
        </h1>
        <p align="center">
            Hi, This is Savatar. Ask me anything !!!
        </p>
        
        <div class="box">
        <div class="chat user">
            <h4>You:</h4>       
            Type here... 
            <br><br>
            <form action = "/predict" method = "POST">
                <input type = "submit" value = "SEND"/>
                <input type="text" class="form-control" name="usrText" id="get_response(user_input)" aria-describedby="emailHelp">
            </form>
        </div>
        </div>
        
        <br>
        
        <div class="box">
        <div class="chat bot">
            <h4>Savatar</h4>
            {{out_put_1}}
        </div>    
        </div>

        <p align="center" style="padding:20px;">v2.0</p>
    </body>
</html>
)

