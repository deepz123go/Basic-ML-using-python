# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 15:02:41 2020

@author: User
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('50_Startups.csv')
x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
"""from sklearn.preprocessing import Imputer
im=Imputer(missing_values='NaN',strategy='mean',axis=0)
im.fit(x[:,1:3])
x[:,1:3]=im.transform(x[:,1:3])"""

#Encoding
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
l=LabelEncoder()
x[:,3]=l.fit_transform(x[:,3])
o=OneHotEncoder(categorical_features=[3])
x=o.fit_transform(x).toarray()
#l1=LabelEncoder()
#y=l1.fit_transform(y)

#remove 1 dummy column
x=x[:,1:]

#include x0=1
x=np.append(arr=np.ones((50,1)).astype(int),values=x,axis=1)

#TEST and TRAINING SET
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

#Feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.fit_transform(x_test)"""

#Trainingset learning
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

#Predicting
y_pred=regressor.predict(x_test)

#Backward elimination
import statsmodels.api as sm
#steby step backwardd
"""x_opt=x[:,[0,3]]
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()"""
#automatic backward
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):                #loop to get max p value and then run the program after eliminating column
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float) #identifying max p value
        if maxVar > sl:                                 #check if max p value > significance
            for j in range(0, numVars - i):             #deleting column of max p value
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = x[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)



#Visualization and comparison
"""plt.scatter(x_train,y_train,color='red')
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('salary vs experience')
plt.xlabel('Years')
plt.ylabel('salary')
plt.show()"""
