
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def svr():

    # Importing the dataset
    dataset = pd.read_csv('Position_Salaries.csv')
    X = dataset.iloc[:, 1:2].values
    y = dataset.iloc[:, 2].values

    # Feature Scaling

    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X = sc_X.fit_transform(X)
    y = sc_y.fit_transform(y)

    # Fitting SVR to the dataset

    regressor = SVR(kernel = 'rbf')
    regressor.fit(X, y)

    # Predicting a new result
    y_pred = regressor.predict(6.5)
    y_pred = sc_y.inverse_transform(y_pred)

    # Visualising the SVR results
    plt.scatter(X, y, color = 'red')
    plt.plot(X, regressor.predict(X), color = 'blue')
    plt.title('Truth or Bluff (SVR)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()

    return "Success"

if __name__=="__main__":

    model_response = svr()

    print('Model response is {}'.format(model_response))
