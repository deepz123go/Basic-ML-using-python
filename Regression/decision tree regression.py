# Regression Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
x1=dataset.iloc[:, 1:2].values
y1=dataset.iloc[:, 2].values
# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
#X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y = sc_y.fit_transform(np.reshape(y,(-1,1)))

# Fitting the Regression Model to the dataset
# Create your regressor here
from sklearn.tree import DecisionTreeRegressor 
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)

# Predicting a new result
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))
# Visualising the Regression results
yp1=sc_y.inverse_transform(regressor.predict(X))
plt.scatter(x1, y1, color = 'red')
#plt.plot(X, regressor.predict(X), color = 'blue')
plt.plot(x1, yp1, color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(x1), max(x1), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
y_grid=sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid)))
plt.plot(X_grid, y_grid, color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()