import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def polynomial_regression():

    # Importing the dataset
    dataset = pd.read_csv('Position_Salaries.csv')
    X = dataset.iloc[:, 1:2].values
    y = dataset.iloc[:, 2].values

    # Fitting Linear Regression to the dataset

    lin_reg = LinearRegression()
    lin_reg.fit(X, y)

    # Fitting Polynomial Regression to the dataset
    poly_reg = PolynomialFeatures(degree = 4)
    X_poly = poly_reg.fit_transform(X)
    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(X_poly, y)

    # Visualising the Polynomial Regression results
    plt.scatter(X, y, color = 'red')
    plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
    plt.title('Truth or Bluff (Polynomial Regression)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()

    # Predicting a new result with Polynomial Regression
    lin_reg_2.predict(poly_reg.fit_transform(6.5))

    return "Success"

if __name__=="__main__":

    model_response = polynomial_regression()

    print('Model Response is {}'.format(model_response))
