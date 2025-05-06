# -*- coding: utf-8 -*-
"""
Created on Mon May  5 13:58:39 2025

@author: spudu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
file_path = r"C:\Users\spudu\Downloads\Housing data 2025.xlsx"
#Excel file reading and manipulating into dataframe
xl = pd.read_excel(file_path)

#%%
#Accessing Necessary Sheets
eth = pd.read_excel(file_path,sheet_name = 'Ethnicity')
eth = eth.drop(eth.index[range(3)])
del eth['Unnamed: 5']
birth = pd.read_excel(file_path,sheet_name = 'Country of birth')
birth = birth.drop(birth.index[range(3)])
del birth['Unnamed: 6']

#%%
#Ethnicity 

reg = eth.iloc[3:12,0:11]

x = np.array([2001, 2006, 2011, 2016])
y = np.array([30,40,50,60,70])

lin_reg = LinearRegression(fit_intercept = True)
x_rs = x.reshape(-1,1)

plt.xticks(x)
plt.yticks(y)
plt.xlabel('Year')
plt.ylabel('Home Ownership(%)')
plt.title('Percentage Home Ownership By Ethnicity [2001-2016]')


for i in range(0,5):
    plt.scatter(x,(eth.iloc[i,5:9])*100,label = eth.iloc[i,0])
    lin_reg.fit(x_rs,(eth.iloc[i,5:9])*100)
    predicted = lin_reg.predict(x_rs)
    plt.plot(x_rs,predicted)


plt.legend(loc = "center right", bbox_to_anchor = (1.5,0.5))
plt.show()

#%%
# Regions
#Accessing sheet related to this
Type = pd.read_excel(file_path,sheet_name = 'Type by Region')
del Type['Unnamed: 6']
Type = Type.rename(columns={'Housing tenure by region, 1996 to 2016':'Region','Unnamed: 1':'1996','Unnamed: 2':'2001','Unnamed: 3':'2006','Unnamed: 4':'2011','Unnamed: 5':'2016','Unnamed: 7':'1996','Unnamed: 8':'2001','Unnamed: 9':'2006','Unnamed: 10':'2011','Unnamed: 11':'2016'})

#Analysis By Region
reg = Type.iloc[3:12,0:11]

x = np.array([1996,2001,2006,2011,2016])
y = np.array([50,55,60,65,70,75])

lin_reg = LinearRegression(fit_intercept = True)
x_rs = x.reshape(-1,1)

p2 = PolynomialFeatures(degree = 2, include_bias = False)
p2I = p2.fit_transform(x.reshape(-1,1))
p2R = LinearRegression(fit_intercept = True)

plt.xticks(x) 
plt.yticks(y) 
plt.xlabel('Year')
plt.ylabel('Home Ownership(%)')
plt.title('Percentage Home Ownership By Region [1996-2016]')

for i in range(0,9):
    plt.scatter(x,(reg.iloc[i,6:11])*100,label = reg.iloc[i,0])
    lin_reg.fit(x_rs,(reg.iloc[i,6:11])*100)
    predicted = lin_reg.predict(x_rs)
    plt.plot(x_rs,predicted)

plt.legend(loc = "center right", bbox_to_anchor = (1.4,0.5))
plt.show()

plt.xticks(x) 
plt.yticks(y) 
plt.xlabel('Year')
plt.ylabel('Home Ownership(%)')
plt.title('Percentage Home Ownership By Region [1996-2016]')

for i in range(0,9):
    plt.scatter(x,(reg.iloc[i,6:11])*100,label = reg.iloc[i,0])
    p2R.fit(p2I,(reg.iloc[i,6:11])*100)
    predict = p2R.predict(p2I)
    plt.plot(x,predict)

plt.legend(loc = "center right", bbox_to_anchor = (1.4,0.5))
plt.show()

#%%
# Age
#Accessing Required Sheet and editing
age = pd.read_excel(file_path,sheet_name = 'Age group')
age = age.drop(age.index[range(3)])
del age['Unnamed: 6']

x = np.array([1996,2001,2006,2011,2016])
y = np.array([10,20,30,40,50,60,70,80])

lin_reg = LinearRegression(fit_intercept = True) 
x_rs = x.reshape(-1,1)

lin_reg = LinearRegression(fit_intercept = True)
x_rs = x.reshape(-1,1)

p2 = PolynomialFeatures(degree = 2, include_bias = False)
p2I = p2.fit_transform(x.reshape(-1,1))
p2R = LinearRegression(fit_intercept = True)

plt.xticks(x) 
plt.yticks(y) 
plt.xlabel('Year')
plt.ylabel('Home Ownership(%)')
plt.title('Percentage Home Ownership By Age [1996-2016]')

for i in range(0,7):
    plt.scatter(x,(age.iloc[i,6:11])*100,label = age.iloc[i,0])
    lin_reg.fit(x_rs,(age.iloc[i,6:11])*100)
    predicted = lin_reg.predict(x_rs)
    plt.plot(x_rs,predicted)

plt.legend(loc = "center right", bbox_to_anchor = (1.4,0.5))
plt.show()

plt.xticks(x) 
plt.yticks(y) 
plt.xlabel('Year')
plt.ylabel('Home Ownership(%)')
plt.title('Percentage Home Ownership By Age [1996-2016]')

for i in range(0,7):
    plt.scatter(x,(age.iloc[i,6:11])*100,label = age.iloc[i,0])
    p2R.fit(p2I,(age.iloc[i,6:11])*100)
    predict = p2R.predict(p2I)
    plt.plot(x,predict)

plt.legend(loc = "center right", bbox_to_anchor = (1.4,0.5))
plt.show()


#%%
# Afforability

#Accessing Necessary Sheets
hp = pd.read_excel(file_path,sheet_name = 'house prices')
hp = hp.drop(hp.index[range(5)])
earn = pd.read_excel(file_path,sheet_name = 'retail prices and earnings')

#calculating average house prices
ahp = []
aearn = []
for k in range(0,43):
    ahp.append((hp.iloc[4*k,1] + hp.iloc[4*k+1,1] + hp.iloc[4*k+2,1] + hp.iloc[4*k+3,1])/4)
    aearn.append(earn.iloc[k+1,2])
    
x = np.array([2000,6000,10000,14000,18000,22000,26000])
y = np.array([10000,50000,90000,130000,170000,210000])

lin_reg = LinearRegression(fit_intercept = True)
x_rs = x.reshape(-1,1)

p2 = PolynomialFeatures(degree = 2, include_bias = False)
p2I = p2.fit_transform(x.reshape(-1,1))
p2R = LinearRegression(fit_intercept = True)

plt.xticks(x) 
plt.yticks(y) 
plt.xlabel('Average Annual Nominal Earnings')
plt.ylabel('Average UK House Price')
plt.title('Comparison of average earnings and house prices over time')

for k in range(0,43):
    plt.scatter(x,(hp.iloc[k,6:11])*100,label = hp.iloc[k,0])
    lin_reg.fit(x_rs,(hp.iloc[k,6:11])*100)
    predicted = lin_reg.predict(x_rs)
    plt.plot(x_rs,predicted)


plt.legend(loc = "center right", bbox_to_anchor = (1.4,0.5))
plt.show()

plt.xticks(x) 
plt.yticks(y) 
plt.xlabel('Average Annual Nominal Earnings')
plt.ylabel('Average UK House Price')
plt.title('Comparison of average earnings and house prices over time')

for k in range(0,43):
    plt.scatter(x,(hp.iloc[k,6:11])*100,label = hp.iloc[k,0])
    p2R.fit(p2I,(hp.iloc[k,6:11])*100)
    predict = p2R.predict(p2I)
    plt.plot(x,predict)

plt.legend(loc = "center right", bbox_to_anchor = (1.4,0.5))
plt.show()

##
x = np.array([1974,1980,1986,1992,1998,2004,2010,2016])
y = np.array([4,5,6,7,8])

lin_reg = LinearRegression(fit_intercept = True)
x_rs = x.reshape(-1,1)

time = []
for l in range(0,43):
    time.append(ahp[l]/aearn[l])



p2 = PolynomialFeatures(degree = 2, include_bias = False)
p2I = p2.fit_transform(x.reshape(-1,1))
p2R = LinearRegression(fit_intercept = True)

plt.xticks(x) 
plt.yticks(y) 
plt.xlabel('Label')
plt.ylabel('Relative House Price')
plt.title('Relative Housing Cost Over Time')

for l in range(0,43):
    plt.scatter(x,(hp.iloc[l,6:11])*100,label = hp.iloc[l,0])
    lin_reg.fit(x_rs,(hp.iloc[l,6:11])*100)
    predicted = lin_reg.predict(x_rs)
    plt.plot(x_rs,predicted)


plt.legend(loc = "center right", bbox_to_anchor = (1.4,0.5))
plt.show()


plt.xticks(x) 
plt.yticks(y) 
plt.xlabel('Label')
plt.ylabel('Relative House Price')
plt.title('Relative Housing Cost Over Time')

for l in range (0,43):
    plt.scatter(x,(hp.iloc[l,6:11])*100,label = hp.iloc[l,0])
    p2R.fit(p2I,(hp.iloc[l,6:11])*100)
    predict = p2R.predict(p2I)
    plt.plot(x,predict)

plt.legend(loc = "center right", bbox_to_anchor = (1.4,0.5))
plt.show()


 





