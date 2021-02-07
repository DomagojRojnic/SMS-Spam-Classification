import numpy as np 
import pandas as pd 
import nltk
import sys
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('punkt')
from nltk.tokenize import word_tokenize

def find_features(message, word_features):
    words = word_tokenize(message)
    features = {}
    for word in word_features:
        features[word] = (word in words)
    return features

#ucitavanje podataka
df = pd.read_table('SMSSpamCollection', header = None, encoding = 'utf-8')

#broj uzoraka pojedine klase
classes = df[0]

#preprocessing

#class labels to binary values, ham=0, spam=1
encode = LabelEncoder()
y = encode.fit_transform(classes)

#store the SMS message data
text_messages = df[1]

# regularni izrazi da zamijenimo email, brojeve, brojeve telefona, url, simbole sa rijecima
processed = text_messages.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$', 'emailaddress')
processed = processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$', 'webaddress')
processed = processed.str.replace(r'£|\$', 'moneysymb')
processed = processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$', 'phonenumbr')
processed = processed.str.replace(r'\d+(\.\d+)?', 'numbr')

# uklanjanje punktacija
processed = processed.str.replace(r'[^\w\d\s]', ' ')

# vise razmaka izmedju rijeci u jedan razmak
processed = processed.str.replace(r'\s+', ' ')

# uklanjanje razmaka na pocetku i kraju poruke
processed = processed.str.replace(r'^\s+|\s+?$', '')

processed = processed.str.lower()

#uklanjanje stop rijeci - rijeci koje ne mijenjaju znacenje recenice
stop_words = set(stopwords.words('english'))
processed = processed.apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))

ps = nltk.PorterStemmer()
processed = processed.apply(lambda x: ' '.join(ps.stem(term) for term in x.split()))

all_words = []
for message in processed:
    words = word_tokenize(message)
    for word in words:
        all_words.append(word)

all_words = nltk.FreqDist(all_words)

#make features
word_features = list(i[0] for i in all_words.most_common(1500))

messages = list(zip(processed, y))

seed = 1
np.random.seed = seed
np.random.shuffle(messages)

# find_features za svaki SMS
feature_sets = [(find_features(text, word_features), label) for (text, label) in messages]

#train test split
training, testing = train_test_split(feature_sets, test_size=0.25, random_state = seed)