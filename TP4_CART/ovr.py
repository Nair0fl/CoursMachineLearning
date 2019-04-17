import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import itertools
import operator


class OvRPredictor:

    def __init__(self,input, target, test_size=0.2):
        self.x_train,self.x_test, self.y_train,  self.y_test = train_test_split(input,target, test_size=test_size)
        self.test_values = [(self.x_test[index],value) for index, value in enumerate(self.y_test)]
        self.classes = set(target)


    def _generateOvRClassifier(self):
        o_vs_r_classifiers = {}
        for elem in self.classes:
            class_valid = [self.x_train[index] for index, value in enumerate(self.y_train) if value == elem]
            class_invalid = [self.x_train[index] for index, value in enumerate(self.y_train) if value != elem]
            value = [1] * len(class_valid) + [0] * len(class_invalid)
            learn = class_valid + class_invalid
            o_vs_r_classifiers["%d_rest" % elem] = LogisticRegression(multi_class='ovr',solver='lbfgs').fit(learn, value)
        return o_vs_r_classifiers


    def _predictOVR(self, o_vs_r_classifiers):
        results = {}
        stats = {'TP' :0, 'FP' : 0, 'TN': 0, 'FN': 0}
        i=0
        for elem in self.test_values:
            intern_result = {}
            for name, classifier in o_vs_r_classifiers.items():
                result = classifier.predict([elem[0]])
                result_proba = classifier.predict_proba([elem[0]])
                intern_result[name.split('_')[0]] = result_proba[0][1]
                if result == 0:
                    if int(name.split('_')[0])!= elem[1]:
                        stats['TN'] += 1
                    else:
                        stats['FN'] +=1
                else:
                    if int(name.split('_')[0])!= elem[1]:
                        stats['FP'] += 1
                    else:
                        stats['TP'] +=1
            results[i] = intern_result
            i+=1
        correct = 0
        for key, elem in results.items():
            predicted = max(elem.items(), key=operator.itemgetter(1))[0]
            value = self.test_values[key][1]
            if int(predicted) == value:
                correct +=1
        
        correctness = (correct/len(results)*100)
        precision = (stats['TP']/(stats['TP'] + stats['FP'])) *100
        recall = (stats['TP']/(stats['TP'] +stats['FN']))*100
        f_measure = ((precision * recall)/(precision + recall))*2
        return {'correctness' :  correctness, 'precision' : precision,
                'recall' : recall, 'f1' : f_measure}

    def run_predict(self):
        ovr_predict = self._generateOvRClassifier()
        return self._predictOVR(ovr_predict)