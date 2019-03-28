import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(0)
x = [[1],[1],[2]]
y =[[1],[2],[2]]


regression_model = LinearRegression()
regression_model.fit(x, y)
y_predicted = regression_model.predict(x)

rmse = mean_squared_error(y, y_predicted)
r2 = r2_score(y, y_predicted)



print('Slope:' ,regression_model.coef_)
print('Intercept:', regression_model.intercept_)
print('Root mean squared error: ', rmse)
print('R2 score: ', r2)
print(np.linalg.norm(y - y_predicted) ** 2)

plt.plot(np.linalg.norm(y - y_predicted) ** 2, color='g')

plt.scatter(x, y, s=10)
plt.scatter(x,y_predicted,s=10)
plt.xlabel('x')
plt.ylabel('y')

plt.plot(x, y_predicted, color='r')
plt.show()