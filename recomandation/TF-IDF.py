from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pa
import os, sys

corpus=[]
learn_path="data/SimpleText/SimpleText_auto//"
dirs = os.listdir( learn_path )
for file in dirs:
    with open(learn_path+file, 'r',encoding="utf8") as fileRead:
        corpus.append(fileRead.read().replace('\n', ''))

vectorizer = TfidfVectorizer()
valeurs = vectorizer.fit_transform(corpus)
liste = vectorizer.get_feature_names()


print(valeurs)