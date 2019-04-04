import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(0)
x = [[0], [2], [4], [8], [9], [10], [11], [12], [14], [19]]
y =[[81682.0], [81720.0], [81760.0], [81826.0], [81844.0], [81864.0], [81881.0], [81900.0], [81933.0], [82003.0]]


regression_model = LinearRegression()
regression_model.fit(x, y)
y_predicted = regression_model.predict(x)
y_lendement=regression_model.predict([[20]])

plt.ylim((81600,82100))

print ("Consomation du lendemain est de "+str(y_lendement[0][0]))

plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x, y_predicted, color='r')
plt.show()