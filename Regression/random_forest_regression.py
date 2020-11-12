import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def random_forest_regression():

    # Importing the dataset
    dataset = pd.read_csv('Position_Salaries.csv')
    X = dataset.iloc[:, 1:2].values
    y = dataset.iloc[:, 2].values

    # Fitting Random Forest Regression to the dataset

    regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
    regressor.fit(X, y)

    # Predicting a new result
    y_pred = regressor.predict(6.5)

    # Visualising the Random Forest Regression results (higher resolution)
    X_grid = np.arange(min(X), max(X), 0.01)
    X_grid = X_grid.reshape((len(X_grid), 1))
    plt.scatter(X, y, color = 'red')
    plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
    plt.title('Truth or Bluff (Random Forest Regression)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()

    return "Success"

if __name__=="__main__":

    model_response = random_forest_regression()

    print('Model Response is {}'.format(model_response))
