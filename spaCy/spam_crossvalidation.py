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


from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

X=doc_vectors
y=data.Category

test_sizes=[0.5,0.6,0.7,0.8,0.9]

for size in test_sizes:
    
    print(f"-----------test_size:{size}-----------")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=1,
                                                        stratify=y, shuffle=True)
    svc = LinearSVC(random_state=1, dual=False, max_iter=10000)
    svc.fit(X_train,y_train)
    
    print(f"Model score: {svc.score(X_test,y_test)*100:.3f}%")
    
    skf = StratifiedKFold(n_splits=10)
    skf_splits = skf.get_n_splits(X_train, y_train)
    
    skf_scores=[]
    
    for train_index, test_index in skf.split(X_train, y_train):
        X_train_fold , X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        svc.fit(X_train_fold, y_train_fold) 
        skf_scores.append(svc.score(X_test_fold,y_test_fold))
    
    cv_scores_macro = cross_val_score(svc, X_train, y_train, cv=skf_splits, scoring='f1_macro')
    cv_scores_micro = cross_val_score(svc, X_train, y_train, cv=skf_splits, scoring='f1_micro')
    
    print(f"Cross-validation f1-macro score: {np.mean(cv_scores_macro) * 100:.3f}%", )
    print(f"Cross-validation f1-micro score: {np.mean(cv_scores_micro) * 100:.3f}%", )
    print(f"Stratified k-fold score: {np.mean(skf_scores) * 100:.3f}%\n", )
