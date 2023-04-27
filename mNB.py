import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import re
from textblob import TextBlob
tweets_df2 = pd.read_csv('C:\\Users\\tolga\\OneDrive\\Masaüstü\\tw2.csv')
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
   return  TextBlob(text).sentiment.polarity


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


def preprocess_data(data):
    # Convert text to lowercase
    data['Text'] = data['Text'].str.strip().str.lower()
    return data

tweets_df2 = preprocess_data(tweets_df2)

x = tweets_df2['Text']
y = tweets_df2['Analysis']
x, x_test, y, y_test = train_test_split(x,y, stratify=y, test_size=0.3, random_state=2000000)

# Vectorize text reviews to numbers
vec = CountVectorizer(stop_words='english')
x = vec.fit_transform(x).toarray()
x_test = vec.transform(x_test).toarray()



model = MultinomialNB()
model.fit(x, y)


y_pred = model.predict(x_test) #getting predictions on the trained model
CR=confusion_matrix(y_test, y_pred)
print(CR)
accuracy=(model.score(x_test, y_test))
print(accuracy)
print(classification_report(y_test,y_pred))

