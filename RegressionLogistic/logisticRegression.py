import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

import itertools
import operator

digits = datasets.load_digits()

data = digits['data']
target  = digits['target']

classes = set(target)


x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)
logisticRegr.predict(x_test[0].reshape(1,-1))
predictions = logisticRegr.predict(x_test)
score = logisticRegr.score(x_test, y_test)
print(score)
cm = metrics.confusion_matrix(y_test, predictions)
print(cm)