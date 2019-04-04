#Problème de invalid value encountered in double_scalars
import numpy as np
import matplotlib.pyplot as plt

points = np.array([[1.0, 81682.0], [3.0, 81720.0], [5.0, 81760.0], [9.0, 81826.0], [10.0, 81844.0], [11.0, 81864.0], [12.0, 81881.0], [13.0, 81900.0], [15.0, 81933.0], [18.0, 82003.0]])
teta = np.array([10., 10.])
features = np.array([])
labels = np.array([])
xModel = np.array([1, 3, 5, 9,10,11,12,13,15])
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
print ("2019-04-05 il y aura : "+str(h(19)))
#Affichage points et model
plt.scatter(features, labels, color="blue")
plt.plot(xModel, yModel, color="red")
axes = plt.axes()
axes.grid()
plt.show()