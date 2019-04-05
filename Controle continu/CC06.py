import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def affichageGraph():
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
    
def affichageRegression(x,y):
    plt.plot(x, y, color='r',label='Regression lineaire')

def affichagePoint(x,y,label,color):
    plt.scatter(x,y,label=label,c=color)
    
def predict(model,x):
    return model.predict(x)

def model(x,y):
    regression_model = LinearRegression()
    regression_model.fit(x, y)
    return regression_model

def main():   
    x = [[0], [2], [4], [8], [9], [10], [11], [12], [14], [19]]
    y =[[81682.0], [81720.0], [81760.0], [81826.0], [81844.0], [81864.0], [81881.0], [81900.0], [81933.0], [82003.0]]
    
    modelRegression=model(x,y)
    x_lendemain=[[20]]
    y_lendemain=predict(modelRegression,x_lendemain)
    y_predict=predict(modelRegression,x)

    
    print ("Consomation du lendemain est de "+str(y_lendemain[0][0]))
    #Affichage du points de prédiction
    #Affichage des points
    affichagePoint(x,y,'Point entrainement','b')
    affichagePoint(x_lendemain,y_lendemain,'Point lendemain','g')
    affichageRegression(x,y_predict)
    affichageGraph()


if __name__ == "__main__":
    main()