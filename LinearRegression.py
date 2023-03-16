#Import the Libraries
import pandas as pd
import matplotlib.pyplot as plt


#Reading Data from Files (Advertizing.csv)
data=pd.read_csv('advertising.csv')
#print(data.head())

#Visualizing the Dataset

fig,axis = plt.subplots(1,3,sharey=True)
data.plot(kind='scatter',x='TV',y='Sales',ax=axis[0],figsize=(16,8))
data.plot(kind='scatter',x='Radio',y='Sales',ax=axis[1],figsize=(16,8))
data.plot(kind='scatter',x='Newspaper',y='Sales',ax=axis[2],figsize=(16,8))

#Linear Regression Variables
feature_cols=['TV']
X=data[feature_cols]
y=data.Sales


#Importing Linear Regression Models
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X,y)

print(lr.intercept_)
print(lr.coef_)

#Using the Slope Equation
#val=int(input())
val=50
result=6.974821488229891+0.05546477*val
print(result)

#Creating a Dataframe with Max and Min Values
x_new=pd.DataFrame({'TV':[data.TV.min(),data.TV.max()]})
print(x_new.head())

preds=lr.predict(x_new)
print(preds)

data.plot(kind='scatter',x='TV',y='Sales')
plt.plot(x_new,preds,c='red',linewidth=3)

import statsmodels.formula.api as smf
lm=smf.ols(formula='Sales ~ TV',data=data).fit()
print(lm.conf_int())

#Finding the Porbability Values
print("Probability Values: ",lm.pvalues)

#finding the RSquared Values
print("R-Squared Values: ",lm.rsquared,end='\n\n\n')


#Multi Linear Regression
feature_cols=['TV','Radio','Newspaper']
X=data[feature_cols]
y=data.Sales


#Importing Linear Regression Models
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X,y)

print(lr.intercept_)
print(lr.coef_)

lm=smf.ols(formula='Sales ~ TV+Radio+Newspaper',data=data).fit()
print(lm.conf_int())
print(lm.summary())
