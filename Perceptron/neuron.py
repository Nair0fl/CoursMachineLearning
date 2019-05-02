import timeit
import functools
from itertools import combinations
import operator
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import random
from functools import reduce
import cv2
import numpy as np

def creer_dataset():
    digits = datasets.load_digits()
    data = digits['data']
    target = digits['target']
    return train_test_split(data, target, test_size=0.3)

def create_neuron():
    neuron = []
    for elem in range(0, 64):    
        neuron.append(random.uniform(0, 1))
    return neuron

def create_network():
    network = []
    for elem in range(0,10):
        neuron=create_neuron()
        network.append(neuron)
    return network

def get_weight(elem,comparison):
    return np.dot(elem, comparison)

def predire(images, network):
    for index, image in enumerate(images):
        scores = [get_weight(elem,image)
                  for elem in network]
        predicted = scores.index(max(scores))
        return predicted
        
def entrainement(images,resultats,network):
    for index, image in enumerate(images):
       predicted=predire([image],network)
       if(predicted!=resultats[index]):
           network[resultats[index]]=set_poids(network[resultats[index]],image)
    return network

def set_poids(neuron,image):
    for index,pixel in enumerate(image):
        if pixel>0 and neuron[index]<255:
            neuron[index]+=1
        elif neuron[index]>0:
            neuron[index]-=1
    return neuron


def main():    
    x_train, x_test, y_train, y_test = creer_dataset()
    network=create_network()
    network=entrainement(x_train,y_train,network)
    correct_prediction=0
    number_of_tests=0
    for index,test in enumerate(x_test):
        a=predire([test],network)
        print(f"prediction : {a}")
        print(f"resultat : {y_test[index]}")

        if a==y_test[index]:
            correct_prediction+=1
        number_of_tests+=1
    print(f"Predict percent : {(correct_prediction/number_of_tests) * 100}%")

if __name__ == "__main__":
	main()