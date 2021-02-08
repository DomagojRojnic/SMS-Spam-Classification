import numpy as np 
import pandas as pd 
import nltk
import sys
import sklearn
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


X=feature_sets

#train test split

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = seed)

svc = LinearSVC(random_state=388, dual=False, max_iter=10000)
svc.fit(X_train,y_train)

print(f"Model score: {svc.score(X_test,y_test)*100:.3f}%")

skf = StratifiedKFold(n_splits=10)
skf_splits = skf.get_n_splits(X_train, y_train)

skf_scores=[]


for train_index, test_index in skf.split(X_train, y_train):
    X_train_fold , X_test_fold = X_train[train_index], X_train[test_index]
    y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
    svc.fit(X_train_fold, y_train_fold) 
    skf_scores.append(svc.score(X_test_fold,y_test_fold))

cv_scores_macro = cross_val_score(svc, X, y, cv=skf_splits, scoring='f1_macro')
cv_scores_micro = cross_val_score(svc, X, y, cv=skf_splits, scoring='f1_micro')

print(f"Cross-validation f1-macro score: {np.mean(cv_scores_macro) * 100:.3f}%", )
print(f"Cross-validation f1-micro score: {np.mean(cv_scores_micro) * 100:.3f}%", )
print(f"Stratified k-fold score: {np.mean(skf_scores) * 100:.3f}%\n", )


