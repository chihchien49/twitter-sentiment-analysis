#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

# This is for making some large tweets to be displayed
pd.options.display.max_colwidth = 100

# I got some encoding issue, I didn't knew which one to use !
# This post suggested an encoding that worked!
# https://stackoverflow.com/questions/19699367/unicodedecodeerror-utf-8-codec-cant-decode-byte
train_data = pd.read_csv("train1.csv", encoding='ISO-8859-1')
train_data


# In[2]:


rand = np.random.randint(1,len(train_data),25).tolist() #全部檔案隨機取50筆資料
train_data["SentimentText"][rand]


# In[3]:


import re
text = train_data.SentimentText.str.cat()
emoji = set(re.findall(r" ([xX:;][-']?.) ",text))
emoji_count = []
for e in emoji:
    emoji_count.append((text.count(e), e))
sorted(emoji_count,reverse=True)


# In[4]:


happy = r" ([xX;:]-?[dD)]|:-?[\)]|[;:][pP]) "
sad = r" (:'?[/|\(]) "
print("Happy emoji icons:", set(re.findall(happy, text)))
print("Sad emoji icons:", set(re.findall(sad, text)))


# In[5]:


import nltk
from nltk.tokenize import word_tokenize

# Uncomment this line if you haven't downloaded punkt before
# or just run it as it is and uncomment it if you got an error.
#nltk.download('punkt')
def most_used_words(text):
    tokens = word_tokenize(text)
    frequency_dist = nltk.FreqDist(tokens)
    print("There is %d different words" % len(set(tokens)))
    return sorted(frequency_dist,key=frequency_dist.__getitem__, reverse=True)


# In[ ]:





# In[6]:


from nltk.corpus import stopwords

#nltk.download("stopwords")

mword = most_used_words(train_data.SentimentText.str.cat())
themost = []
for w in mword:
    if len(themost) == 1000:
        break
    if w in stopwords.words("english"):
        continue
    else:
        themost.append(w)


# In[7]:


sorted(themost)


# In[8]:


from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

#nltk.download('wordnet')
def stem_tokenize(text):
    stemmer = SnowballStemmer("english")
    stemmer = WordNetLemmatizer()
    return [stemmer.lemmatize(token) for token in word_tokenize(text)]

def lemmatize_tokenize(text):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in word_tokenize(text)]


# In[9]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline

class TextPreProc(BaseEstimator,TransformerMixin):
    def __init__(self, use_mention=False):
        self.use_mention = use_mention
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        # We can choose between keeping the mentions
        # or deleting them
        if self.use_mention:
            X = X.str.replace(r"@[a-zA-Z0-9_]* ", " @tags ")
        else:
            X = X.str.replace(r"@[a-zA-Z0-9_]* ", "")
            
        # Keeping only the word after the #
        X = X.str.replace("#", "")
        X = X.str.replace(r"[-\.\n]", "")
        # Removing HTML garbage
        X = X.str.replace(r"&\w+;", "")
        # Removing links
        X = X.str.replace(r"https?://\S*", "")
        # replace repeated letters with only two occurences
        # heeeelllloooo => heelloo
        X = X.str.replace(r"(.)\1+", r"\1\1")
        # mark emoticons as happy or sad
        X = X.str.replace(happy, " happyemoticons ")
        X = X.str.replace(sad, " sademoticons ")
        X = X.str.lower()
        return X


# In[10]:


from sklearn.model_selection import train_test_split
import nltk
nltk.download('wordnet')
sentiments = train_data['Sentiment']
text = train_data['SentimentText']

# I get those parameters from the 'Fine tune the model' part
vectorizer = TfidfVectorizer(tokenizer=lemmatize_tokenize, ngram_range=(1,2))
pipeline = Pipeline([
    ('text_pre_processing', TextPreProc(use_mention=True)),
    ('vectorizer', vectorizer),
])


x_train, x_test, y_train, y_test = train_test_split(text, sentiments, test_size=0.25)


# In[11]:


from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import matplotlib.pyplot as plt

token = Tokenizer(num_words=3800) 
token.fit_on_texts(x_train)
x_train_seq = token.texts_to_sequences(x_train)
x_test_seq = token.texts_to_sequences(x_test)
x_train = sequence.pad_sequences(x_train_seq, maxlen=380)
x_test = sequence.pad_sequences(x_test_seq, maxlen=380)


modelLSTM = Sequential() #建立模型
modelLSTM.add(Embedding(output_dim=32,input_dim=3800,input_length=380)) 

modelLSTM.add(Dropout(0.2)) #隨機在神經網路中放棄20%的神經元，避免overfitting
modelLSTM.add(LSTM(32)) 
modelLSTM.add(Dense(units=256,activation='relu')) 
#建立256個神經元的隱藏層
modelLSTM .add(Dropout(0.2))
modelLSTM .add(Dense(units=1,activation='sigmoid'))
 #建立一個神經元的輸出層
modelLSTM.summary()
modelLSTM.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy']) 
train_history=modelLSTM.fit(x_train,y_train,epochs=10, batch_size=100,verbose=2,validation_split=0.2)
scores = modelLSTM .evaluate(x_test, y_test,verbose=1)
print("Accuracy:"+str(scores[1]))
plt.plot(train_history.history['accuracy'])
plt.plot(train_history.history['loss'])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




