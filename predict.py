import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LinearRegression, Lasso

from sklearn.svm import SVR
import matplotlib.pyplot as plt

class Predict(object):
    def __init__(self, data):
        self.data = data 
        self.X = self.data.drop(['quality','type','grade'], axis=1)
        self.y = self.data['quality']
    def evaulate(self, y_test, y_pred):
        # Evaluate the model's performance
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print("Mean squared error: ", mse)
        print("R-squared: ", r2)



        plt.scatter(range(len(y_test)), y_test,
                    color='blue', label='True Score')
        plt.scatter(range(len(y_pred)), y_pred,
                    color='red', label='Predicted Score')
        plt.xlabel('Index')
        plt.ylabel('Score')
        plt.legend()
        plt.title('Predicted Score vs. True Score')
        plt.show()
    def RanForReg(self):
        df = self.data
        X = self.X
        y = self.y

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Create the RandomForestRegressor model
        model = RandomForestRegressor(n_estimators=100, random_state=42)

        # Train the model on the training dataset
        model.fit(X_train, y_train)

        # Make predictions on the testing dataset
        y_pred = model.predict(X_test)
        return self.evaulate(y_test,y_pred)

    def RanForRegNorm(self):
        df = self.data
        X = StandardScaler().fit_transform(self.X)
        y = self.y

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=0)

        # Create the RandomForestRegressor model
        model = RandomForestRegressor(n_estimators=100, random_state=42)

        # Train the model on the training dataset
        model.fit(X_train, y_train)

        # Make predictions on the testing dataset
        y_pred = model.predict(X_test)
        return self.evaulate(y_test, y_pred)
    def RidgeReg(self):
        df = self.data
        X = self.X
        y = self.y
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Create the Ridge Regression model
        model = Ridge(alpha=1.0)

        # Train the model on the training dataset
        model.fit(X_train, y_train)

        # Make predictions on the testing dataset
        y_pred = model.predict(X_test)
        return self.evaulate(y_test, y_pred)
       
    def LinReg(self):
        df = self.data
        X = self.X
        y = self.y
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Create the Ridge Regression model
        model = LinearRegression()

        # Train the model on the training dataset
        model.fit(X_train, y_train)

        # Make predictions on the testing dataset
        y_pred = model.predict(X_test)
        return self.evaulate(y_test, y_pred)
       

    def SupportVectorReg(self, kernel):
        df = self.data
        X = self.X
        y = self.y
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Create the Ridge Regression model
        model = SVR(kernel=kernel)

        # Train the model on the training dataset
        model.fit(X_train, y_train)

        # Make predictions on the testing dataset
        y_pred = model.predict(X_test)
        return self.evaulate(y_test, y_pred)
       
    def LassoReg(self):
        df = self.data
        X = self.X
        y = self.y
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Create the Ridge Regression model
        model = Lasso(alpha=1.0)

        # Train the model on the training dataset
        model.fit(X_train, y_train)

        # Make predictions on the testing dataset
        y_pred = model.predict(X_test)
        return self.evaulate(y_test, y_pred)
