import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as sm
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

def multi_linear_regression():

    # Importing the dataset
    dataset = pd.read_csv('50_Startups.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 4].values

    # Encoding categorical data

    labelencoder = LabelEncoder()
    X[:, 3] = labelencoder.fit_transform(X[:, 3])
    onehotencoder = OneHotEncoder(categorical_features = [3])
    X = onehotencoder.fit_transform(X).toarray()

    # Avoiding the Dummy Variable Trap
    X = X[:, 1:]

    # Splitting the dataset into the Training set and Test set

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


    # Fitting Multiple Linear Regression to the Training set

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = regressor.predict(X_test)

    return "Success"

def feature_selection():

    # Building the optimal model using Backward Elimination
    # Feature selection

    # Importing the dataset
    dataset = pd.read_csv('50_Startups.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 4].values
    X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
    X_opt = X[:, [0, 1, 2, 3, 4, 5]]
    regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
    regressor_OLS.summary()
    X_opt = X[:, [0, 1, 3, 4, 5]]
    regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
    regressor_OLS.summary()
    X_opt = X[:, [0, 3, 4, 5]]
    regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
    regressor_OLS.summary()
    X_opt = X[:, [0, 3, 5]]
    regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
    regressor_OLS.summary()
    X_opt = X[:, [0, 3]]
    regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
    regressor_OLS.summary()

    return "Success"


if __name__=="__main__":

    model_response = multi_linear_regression()

    print('Model Response is {}'.format(model_response))

    feature_selection_response = feature_selection()

    pritn('Feature Selection Response is {}'.format(feature_selection_response))