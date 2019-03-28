import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

points = np.array([[1, 1], [1, 2], [2, 2]])
teta = np.array([5., 5.])
features = np.array([])
labels = np.array([])
xModel = np.array([0, 1, 2, 3])
yModel = np.array([])
m = points.size
alpha = 2

def h(x):
    return teta[0] + teta[1] * x

def J():
    somme = 0
    for point in points:
        somme += (h(point[0]) - point[1]) ** 2
    return (1/(2*m)) * somme

def new_Teta0():
    somme = 0
    for point in points:
        somme += h(point[0]) - point[1]
    return teta[0] - (alpha / m) * somme

def new_Teta1():
    somme = 0
    for point in points:
        somme += (h(point[0]) - point[1]) * point[0]
    return teta[1] - (alpha / m) * somme

#Régression linéaire
valeurJ = 0
for n in range(1, 1000):
    alpha = m/n
    tempTeta0 = new_Teta0()
    tempTeta1 = new_Teta1()
    teta[0] = tempTeta0
    teta[1] = tempTeta1

#Séparation des X et des Y
for point in points:
    features = np.append(features, point[0])
    labels = np.append(labels, point[1])

#Calcul des valeurs theoriques
for x in xModel:
    yModel = np.append(yModel, h(x))
    
print(teta)
print(J())

#Affichage points et model
plt.scatter(features, labels, color="blue")
plt.plot(xModel, yModel, color="red")
axes = plt.axes()
axes.grid()
plt.show()