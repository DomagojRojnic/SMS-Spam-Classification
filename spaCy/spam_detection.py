# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 18:11:06 2021

@author: Jakov
"""

import pandas as pd
import spacy
import numpy as np

data = pd.read_csv("spam_data.csv")

spam = data
nlp = spacy.load("en_core_web_lg")

with nlp.disable_pipes():
    doc_vectors = np.array([nlp(text).vector for text in spam.Message])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(doc_vectors, spam.Category,
                                                    stratify=spam.Category, test_size=0.8, random_state=388)

from sklearn.svm import LinearSVC

svc = LinearSVC(random_state=388, dual=False, max_iter=10000)
svc.fit(X_train, y_train)
print(f"Accuracy: {svc.score(X_test, y_test) * 100:.3f}%", )
