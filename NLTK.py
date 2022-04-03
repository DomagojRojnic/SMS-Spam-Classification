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

#import data
df = pd.read_table('SMSSpamCollection', header = None, encoding = 'utf-8')

#samples per class
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

# remove punctuations
processed = processed.str.replace(r'[^\w\d\s]', ' ')

# combine multiple spaces to single
processed = processed.str.replace(r'\s+', ' ')

# remove spaces before and after the word
processed = processed.str.replace(r'^\s+|\s+?$', '')

processed = processed.str.lower()

# remove stop-words
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

# find_features for each SMS
feature_sets = [(find_features(text, word_features), label) for (text, label) in messages]


from sklearn import model_selection

# split the data into training and testing datasets
training, testing = model_selection.train_test_split(feature_sets, test_size = 0.25, random_state=seed)

# We can use sklearn algorithms in NLTK
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Define models to train
names = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "SGD Classifier",
         "Naive Bayes", "SVM Linear"]

classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter = 100),
    MultinomialNB(),
    SVC(kernel = 'linear')
]

models = zip(names, classifiers)

for name, model in models:
    nltk_model = SklearnClassifier(model)
    nltk_model.train(training)
    accuracy = nltk.classify.accuracy(nltk_model, testing)*100
    print("{} Accuracy: {}".format(name, accuracy))
    
from sklearn.ensemble import VotingClassifier

names = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "SGD Classifier",
         "Naive Bayes", "SVM Linear"]

# Removing KNeighbors and Decision Tree due to being too inaccurate
# Combine all models into an ensemble and then evaluate it

classifiers = [
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter = 100),
    MultinomialNB(),
    SVC(kernel = 'linear')
]

models = list(zip(names, classifiers))

nltk_ensemble = SklearnClassifier(VotingClassifier(estimators = models, voting = 'hard', n_jobs = -1))


nltk_ensemble.train(training)
accuracy = nltk.classify.accuracy(nltk_model, testing)*100
print("Voting Classifier: Accuracy: {}".format(accuracy))

# make class label prediction for testing set
txt_features, labels = zip(*testing)

prediction = nltk_ensemble.classify_many(txt_features)

# print a confusion matrix and a classification report
print(classification_report(labels, prediction))

pd.DataFrame(
    confusion_matrix(labels, prediction),
    index = [['actual', 'actual'], ['ham', 'spam']],
    columns = [['predicted', 'predicted'], ['ham', 'spam']])


import joblib

filename="ensemble_model.sav"
joblib.dump(nltk_ensemble, filename)
