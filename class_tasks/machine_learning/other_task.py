# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt 
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error

# data = pd.read_csv('student_performance.csv')
# data.info()
# data.head()

# # separating features

# x = data[["study_hours", "attendance"]]
# y = data[["final_score"]]

# # splitting data

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# # initializing model

# model = LinearRegression()

# # fiting data into a model

# model.fit(x_train, y_train)

# plt.scatter(x_test["study_hours"], y_test, color='blue', label='Actual Data')
# plt.xlabel('Study Hours')
# plt.ylabel('Final Score')
# plt.show()


from sklearn.model_selection import RandomizedSearchCV
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score 
from sklearn.model_selection import train_test_split
dataset_url = "https://media.geeksforgeeks.org/\
wp-content/uploads/20240617221743/cars.csv"
df_cars = pd.read_csv(dataset_url)
# Independent variables
X = df_cars[["Year", "Kilometers_Driven", "Mileage", 
 "Engine", "Power", "Seats"]]
# Dependent variable
Y = df_cars["Price"]
# Split the data
X_train, X_test, y_train, y_test = train_test_split(
 X, Y, test_size=0.2, random_state=0)
model = LinearRegression()
param_space = {'copy_X': [True,False], 
 'fit_intercept': [True,False], 
 'n_jobs': [1,5,10,15,None], 
 'positive': [True,False]}
random_search = RandomizedSearchCV(model, param_space, n_iter=100, cv=5)
random_search.fit(X_train, y_train)
# Parameter which gives the best results
print(f"Best Hyperparameters: {random_search.best_params_}")
# Accuracy of the model after using best parameters
print(f"Best Score: {random_search.best_score_}")