from sklearn.model_selection import train_test_split
from sklearn import linear_model
import pandas as pd
# I used linear regression
# loading the data
ds = pd.read_csv('ds_salaries.csv', header=0)

# removing non numeric columns
ds = ds._get_numeric_data()

# features and label
X = ds.iloc[0:]
y = ds.iloc[:, -1]
# training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# model creation
l_reg = linear_model.LinearRegression()
# model training
model = l_reg.fit(X_train, y_train)
# model testing
predictions = model.predict(X_test)

print('predictions:', predictions)
print('R^2 value:', l_reg.score(X, y))  # accuracy
print('coefficient_factors:', l_reg.coef_)  # slopes
print('intercept:', l_reg.intercept_)
print('actual value:', y[1])
print('predicted value:', l_reg.predict(X)[1])










