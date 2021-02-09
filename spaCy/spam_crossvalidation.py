# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 20:06:27 2021

@author: Jakov
"""

import pandas as pd
import spacy
import numpy as np


data = pd.read_csv("spam_data.csv")

nlp = spacy.load("en_core_web_lg")

with nlp.disable_pipes():
    doc_vectors = np.array([nlp(text).vector for text in data.Message])

# Importing models
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# Model evaluation
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


X=doc_vectors
y=data.Category
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=388,
                                                        stratify=y, shuffle=True)
y_train.reset_index(drop=True,inplace=True)


svc = SVC(C= 10, gamma= 0.1, kernel= 'rbf').fit(X_train,y_train)

logreg = LogisticRegression(solver='liblinear',random_state=388).fit(X_train,y_train)

rfc = RandomForestClassifier(bootstrap=False, max_depth=None,min_samples_leaf=1,min_samples_split=2,n_estimators=200).fit(X_train,y_train)

mlp = MLPClassifier(activation = 'relu', solver = 'adam', learning_rate='adaptive', alpha = 0.0001,
                      hidden_layer_sizes=(100,100,100)).fit(X_train,y_train)


models=[svc,logreg,rfc,mlp]

for model in models:

    print(f"------------Model: {model}------------")
    print(f"Model score: {model.score(X_test,y_test)*100:.3f}%")
    
    skf = StratifiedKFold(n_splits=10)
    skf_splits = skf.get_n_splits(X_train, y_train)
    
    skf_scores=[]
    
    
    for train_index, test_index in skf.split(X_train, y_train):
        X_train_fold , X_test_fold = X_train[train_index], X_train[test_index]
        y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
        model.fit(X_train_fold, y_train_fold) 
        skf_scores.append(model.score(X_test_fold,y_test_fold))
    
    cv_scores_macro = cross_val_score(model, X, y, cv=skf_splits, scoring='f1_macro')
    cv_scores_micro = cross_val_score(model, X, y, cv=skf_splits, scoring='f1_micro')
    
    print(f"Cross-validation f1-macro score: {np.mean(cv_scores_macro) * 100:.3f}%", )
    print(f"Cross-validation f1-micro score: {np.mean(cv_scores_micro) * 100:.3f}%", )
    print(f"Stratified k-fold score: {np.mean(skf_scores) * 100:.3f}%\n", )

