import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import itertools
import operator

digits = datasets.load_digits()

data = digits['data']
target  = digits['target']

classes = set(target)


x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

o_vs_o_classifiers = {}
for elem in itertools.combinations(classes,2):
    class0 = [x_train[index] for index, value in enumerate(y_train) if value == elem[0]]
    class1 = [x_train[index] for index, value in enumerate(y_train) if value == elem[1]]
    value = [0] * len(class0) + [1] * len(class1)
    learn = class0 + class1
    o_vs_o_classifiers['%d_%d'%elem] = LogisticRegression(solver='lbfgs').fit(learn, value)

test_values = [(x_test[index],value) for index, value in enumerate(y_test)]


results = {}
i=0
for elem in test_values:
    intern_result = {}
    for name,classifiers in o_vs_o_classifiers.items():
        result = classifiers.predict([elem[0]])
        members = name.split('_')
        if intern_result.get(members[result[0]]):
            intern_result[members[result[0]]] += 1
        else:
            intern_result[members[result[0]]] = 1
    results[i] = intern_result
    i+=1
    

for key,elem in results.items():
    predicted = max(elem.items(), key=operator.itemgetter(1))[0]
    value = test_values[key][1]
    print("Predicted %s and value was %s" %(predicted,value))
