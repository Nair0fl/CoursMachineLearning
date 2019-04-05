'''Regression Lineaire'''
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def affichage_graph():
    """Creer le model a partir des données et de la classe
    """
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

def affichage_regression(x_point, y_point):
    """Creer le model a partir des données et de la classe
    """
    plt.plot(x_point, y_point, color='r', label='Regression lineaire')

def affichage_point(x_point, y_point, label, color):
    """Creer le model a partir des données et de la classe
    """
    plt.scatter(x_point, y_point, label=label, c=color)

def predict(model, x_point):
    """Creer le model a partir des données et de la classe
    """
    return model.predict(x_point)

def create_model(x_point, y_point):
    """Creer le model a partir des données et de la classe
    """
    regression_model = LinearRegression()
    regression_model.fit(x_point, y_point)
    return regression_model

def main():
    """Creer le model a partir des données et de la classe
    """
    list_x = [[0], [2], [4], [8], [9], [10], [11], [12], [14], [19]]
    list_y = [[81682.0], [81720.0], [81760.0], [81826.0], [81844.0], [81864.0],
              [81881.0], [81900.0], [81933.0], [82003.0]]
    model_regression = create_model(list_x, list_y)
    x_lendemain = [[20]]
    y_lendemain = predict(model_regression, x_lendemain)
    y_predict = predict(model_regression, list_x)
    print("Consomation du lendemain est de "+ str(y_lendemain[0][0]))
    #Affichage du points de prédiction
    #Affichage des points
    affichage_point(list_x, list_y, 'Point entrainement', 'b')
    affichage_point(x_lendemain, y_lendemain, 'Point lendemain', 'g')
    affichage_regression(list_x, y_predict)
    affichage_graph()

if __name__ == "__main__":
    main()
    