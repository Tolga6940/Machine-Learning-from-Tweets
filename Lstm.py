import nltk
import sns as sns
from nltk import RegexpTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from numpy import array
from keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from mlxtend.plotting import plot_confusion_matrix
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report, confusion_matrix
from keras.layers.core import Activation, Dropout, Dense
from tensorflow.keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from tensorflow.keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from wordcloud import WordCloud, STOPWORDS
import csv
import re
from textblob import TextBlob
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('always')


tweets_df2 = pd.read_csv('C:\\Users\\tolga\\OneDrive\\Masaüstü\\tw2.csv',skiprows=1)
tweets_df2.columns = ['Text']


def clean(location):
    letters = re.sub("@[A-Za-z0-9]+", "", location)
    letters = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", letters)
    letters = re.sub(',', '', letters)
    letters = re.sub('//', '', letters)
    letters = re.sub('RT[\s]+', '', letters)
    letters = " ".join(letters.split())
    letters = letters.replace("#", "").replace("_", "")
    letters = re.sub('[()!?:|]', ' ', letters)
    letters = re.sub('\[.*?\]', ' ', letters)
    return letters


emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002500-\U00002BEF"  # chinese char
                           u"\U00002702-\U000027B0"
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           u"\U0001f926-\U0001f937"
                           u"\U00010000-\U0010ffff"
                           u"\u2640-\u2642"
                           u"\u2600-\u2B55"
                           u"\u200d"
                           u"\u23cf"
                           u"\u23e9"
                           u"\u231a"
                           u"\ufe0f"  # dingbats
                           u"\u3030"
                           "]+", flags=re.UNICODE)
tweets_df2['Text'] = tweets_df2['Text'].apply(clean).str.replace(emoji_pattern, '')


def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity


def getPolarity(text):
    return TextBlob(text).sentiment.polarity


tweets_df2['Subjectivity'] = tweets_df2['Text'].apply(getSubjectivity)
tweets_df2['Polarity'] = tweets_df2['Text'].apply(getPolarity)


def getAnalysis(score):
    if score < 0:
        return 'negative'
    elif score == 0:
        return 'neutral'
    else:
        return 'positive'



tweets_df2['Analysis'] = tweets_df2['Polarity'].apply(getAnalysis)



stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
                'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before',
                'being', 'below', 'between', 'both', 'by', 'can', 'd', 'did', 'do',
                'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from',
                'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
                'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
                'into', 'is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
                'me', 'more', 'most', 'my', 'myself', 'now', 'o', 'of', 'on', 'once',
                'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'own', 're', 's', 'same', 'she', "shes",
                'should', "shouldve", 'so', 'some', 'such',
                't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
                'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
                'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was',
                'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom',
                'why', 'will', 'with', 'won', 'y', 'you', "youd", "youll", "youre",
                "youve", 'your', 'yours', 'yourself', 'yourselves']

STOPWORDS = set(stopwordlist)


def cleaning_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])


tweets_df2['Text'] = tweets_df2['Text'].apply(lambda text: cleaning_stopwords(text))
tweets_df2['Text'].head()

tokenizer = RegexpTokenizer(r'\w+')
tweets_df2['Text'] = tweets_df2['Text'].apply(tokenizer.tokenize)

st = nltk.PorterStemmer()


def stemming_on_text(data):
    text = [st.stem(word) for word in data]
    return text


tweets_df2['Text'] = tweets_df2['Text'].apply(lambda x: stemming_on_text(x))
tweets_df2['Text'].head()

lm = nltk.WordNetLemmatizer()


def lemmatizer_on_text(data):
    text = [lm.lemmatize(word) for word in data]
    return text


tweets_df2['Text'] = tweets_df2['Text'].apply(lambda x: lemmatizer_on_text(x))
tweets_df2['Text'].head()

X=tweets_df2['Text']
y=tweets_df2['Analysis']
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
y=lb.fit_transform(y)
max_len = 500
tok = Tokenizer(num_words=2000)
tok.fit_on_texts(X)
sequences = tok.texts_to_sequences(X)
sequences_matrix = pad_sequences(sequences,maxlen=max_len)



X_train, X_test, Y_train, Y_test = train_test_split(sequences_matrix, y, test_size=0.3, random_state=2000000)

def tensorflow_based_model(): #Defined tensorflow_based_model function for training tenforflow based model
    inputs = Input(name='inputs',shape=[max_len])#step1
    layer = Embedding(2000,50,input_length=max_len)(inputs) #step2
    layer = LSTM(64)(layer) #step3
    layer = Dense(256,name='FC1')(layer) #step4
    layer = Activation('relu')(layer) # step5
    layer = Dropout(0.5)(layer) # step6
    layer = Dense(1,name='out_layer')(layer) #step4 again but this time its giving only one output as because we need to classify the tweet as positive or negative
    layer = Activation('sigmoid')(layer) #step5 but this time activation function is sigmoid for only one output.
    model = Model(inputs=inputs,outputs=layer) #here we are getting the final output value in the model for classification
    return model #function returning the value when we call it


model = tensorflow_based_model() # here we are calling the function of created model
model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])


history=model.fit(X_train,Y_train,batch_size=80,epochs=1, validation_split=0.1) # here we are starting the training of model by feeding the training data
print('Training finished !!')
accr1 = model.evaluate(X_test,Y_test) #we are starting to test the model here


print('Test set\n  Accuracy: {:0.2f}'.format(accr1[1])) #the accuracy of the model on test data is given below


y_pred = model.predict(X_test) #getting predictions on the trained model
y_pred = (y_pred > 0.5)


print('\n')
print("confusion matrix")
print('\n')
CR=confusion_matrix(Y_test, y_pred)
print(CR)
print('\n')

fig, ax = plot_confusion_matrix(conf_mat=CR,figsize=(10, 10),
                                show_absolute=True,
                                show_normed=True,
                                colorbar=True)
plt.show()


print(classification_report(Y_test,y_pred))