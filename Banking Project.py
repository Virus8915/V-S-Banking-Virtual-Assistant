#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import re
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.pairwise import cosine_similarity


# In[2]:


df1 = pd.read_csv("BankFAQs.csv")
df1.head()


# In[3]:


df1.info()


# In[4]:


df1.isna().sum()


# In[5]:


df1.head()


# In[6]:


df2 = pd.read_csv("Bank_faq.csv")
df2.head()


# In[7]:


df = pd.concat([df1, df2], axis =0)


# In[8]:


df.info()


# Exploratory Data Analysis

# In[9]:


plt.figure(figsize=(20,14))
sb.set(font_scale=1.5)
sb.set_style('whitegrid')
sb.countplot(df['Class'])


# •	Maximum Query arises in Insurance Class while least query comes under Fund Transfer Section.
# •	Cards, Account and Loan Enquiries are also high enough.
# •	Fewer Enquiries can be seen in Investments and Security.

# In[16]:


plt.figure(figsize=(20,14))
df['Class'].value_counts().plot(kind = 'pie', autopct='%1.1f%%')


# In[17]:


26.2+22.6+21.2+18.2


# •	For Insurance 26.2% enquiries, For Cards 22.6% enquiries, For Loans 21.2% enquiries and For Accounts 18.2% enquiries are       there.
# •	Security and Fund Transfer are having combined enquiry of 4.1% only.
# •	More than 88% enquiries are under 4 classes out of 7 classes.
# 

# In[18]:


df.head()


# In[19]:


import string
string.punctuation


# In[20]:


def remove_punc(txt):
    txt_wo_punct = "".join([i for i in txt if i not in string.punctuation])
    return txt_wo_punct


# In[21]:


df['Question'] = df['Question'].apply(lambda x: remove_punc(x))
df['Answer'] = df['Answer'].apply(lambda x: remove_punc(x))
df.head()


# In[22]:


nltk.download('stopwords')


# In[23]:


stop_words = set(stopwords.words('english'))


# In[24]:


nltk.download('wordnet')


# In[25]:


df['Question'][:5]


# In[26]:


wnl = WordNetLemmatizer()


# In[27]:


def clean(content):
    word_tok = nltk.word_tokenize(content)
    cleaned_words = [word for word in word_tok if word not in stop_words]
    cleaned_words = [wnl.lemmatize(word) for word in cleaned_words]
    return ' '.join(cleaned_words)


# In[28]:


questions = df['Question'].values


# In[29]:


X = []

for question in questions:
    X.append(clean(question))


# In[30]:


X[:15]


# In[31]:


tf = TfidfVectorizer()
X = tf.fit_transform(X)


# In[32]:


X.shape


# In[33]:


le = LabelEncoder()
le.fit(df['Class'])


# In[34]:


y = le.fit_transform(df['Class'])


# In[35]:


y.shape


# In[36]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=500)


# In[37]:


model_params= {
    'svm':{
        'model':SVC(gamma='auto'),
        'params':{
            'C': [1,10,20],
            'kernel': ['rbf', 'linear']
        }
    },
    'random_forest':{
        'model':RandomForestClassifier(),
        'params':{
            'n_estimators': [1,5,10]
        }
    },
    'logistic_regression':{
        'model':LogisticRegression(solver='liblinear', multi_class='auto'),
        'params':{
            'C': [1,5,10]
        }
    }
    
}


# In[38]:


scores = []

for model_name, mp in model_params.items():
    clf = GridSearchCV(mp['model'], mp['params'], cv = 5, return_train_score=False)
    clf.fit(X, y)
    scores.append({
        'model': model_name,
        'best_score':clf.best_score_,
        'best_params' :clf.best_params_
    })


# In[39]:


df_best_score = pd.DataFrame(scores, columns=['model','best_score','best_params' ])
df_best_score


# In[40]:


final_model = SVC(kernel='linear', C=1.0, gamma='auto'  )
final_model.fit(X_train, y_train)


# In[41]:


final_model.score(X_test, y_test)


# In[42]:


class_=le.inverse_transform(final_model.predict(X))


# In[43]:


class_


# In[44]:


import numpy as np


# In[45]:


final_model.predict(X_test)


# In[46]:


from sklearn.metrics.pairwise import cosine_similarity
user_question = "How to open savings account"
## Create a TF-IDF vectorizer to convert the text data and query to a vector representation


# Get the vector representationthe question and answer
answer_vectors = tf.transform(df['Answer']).toarray()
test_vector = tf.transform([user_question]).toarray()
# Calculate the cosine similarity between both vectors
cosine_sims = cosine_similarity(answer_vectors, test_vector)
# Get the index of the most similar text to the query
most_similar_idx = np.argmax(cosine_sims)
# Print the most similar text as the answer to the query
print(df.iloc[most_similar_idx]['Answer'])


# In[62]:



def get_response(usrText):

    while True:

        if usrText.lower() == "bye":
            return "Bye"

        GREETING_INPUTS = ["hello", "hi", "greetings", "sup", "what's up", "hey","hiii","hii","yo"]

        a = [x.lower() for x in GREETING_INPUTS]

        sd=["Thanks","Welcome"]

        d = [x.lower() for x in sd]


        am=["OK"]

        c = [x.lower() for x in am]

        t_usr = tf.transform([clean(usrText.strip().lower())])
        class_ = le.inverse_transform(final_model.predict(t_usr))

        questionset = df[df['Class'].values == class_]

        cos_sims = []
        for question in questionset['Question']:
            sims = cosine_similarity(tf.transform([question]), t_usr)

            cos_sims.append(sims)

        ind = cos_sims.index(max(cos_sims))

        b = [questionset.index[ind]]

        if usrText.lower() in a:

            return ("Hi, I'm Emily!\U0001F60A")


        if usrText.lower() in c:
            return "Ok...Alright!\U0001F64C"

        if usrText.lower() in d:
            return ("My pleasure! \U0001F607")

        if max(cos_sims) > [[0.]]:
            a = df['Answer'][questionset.index[ind]]+"   "
            return a


        elif max(cos_sims)==[[0.]]:
           return "sorry! \U0001F605"


# In[63]:


from flask import Flask, render_template, request, redirect
app = Flask(__name__)


# In[ ]:


@app.route('/')
def index1():
    return render_template('index1.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    out_put_1 = []
    if request.method == 'POST':
        user_text = request.form['usrText']
        out_put= get_response(user_text)
        out_put_1.append(out_put)
    return render_template('index1.html', out_put_1 = out_put_1)


if __name__ == '__main__':
    app.run()


# In[ ]:





# In[ ]:





# In[ ]:




