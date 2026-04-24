import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


data = pd.read_csv('my_dataset.csv')

print("Missing values before cleaning:\n", data.isnull().sum())
data = data.dropna()  # Simple approach: remove rows with any missing values
print("Total rows after handling missing values:", len(data))

Q1 = data.select_dtypes(include=[np.number]).quantile(0.25)
Q3 = data.select_dtypes(include=[np.number]).quantile(0.75)
IQR = Q3 - Q1

data = data[~((data.select_dtypes(include=[np.number]) < (Q1 - 1.5 * IQR)) | 
             (data.select_dtypes(include=[np.number]) > (Q3 + 1.5 * IQR))).any(axis=1)]

# convert categorical variables to dummy/one-hot encoded columns
# we only need the encoded area_type columns as features for this example
# keep other dummies in case you want to expand later

data = pd.get_dummies(data, drop_first=True)

# after encoding, the original "area_type" column is removed and replaced by
# one or more dummy columns named like "area_type_<value>".  select those
# for our feature matrix.  if you only want a single numeric representation
# you could instead map categories to codes before encoding.
area_cols = [col for col in data.columns if col.startswith('area_type_')]
if not area_cols:
    raise KeyError("No area_type dummy columns found after get_dummies")

X = data[area_cols]
# target remains price
y = data['price']   

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# if X_test has multiple columns, use the row index for plotting
if X_test.shape[1] == 1:
    plt.scatter(X_test, y_test, color='blue', label='Actual Data')
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
    plt.xlabel('Feature')
else:
    # fall back to plotting actual vs predicted to visualize fit
    plt.scatter(y_test, y_pred, color='blue', label='Actual vs Predicted')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             color='red', linewidth=2, label='Ideal')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')

plt.title('Linear Regression Model')
plt.legend()
plt.show()
