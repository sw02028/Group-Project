# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 10:43:49 2025

@author: FiercePC
"""
#PART B :- HOME OWNERSHIP DATA ANALYSIS
#Affordability of Housing

#Importing Libraries
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

#Excel file reading and manipulating into dataframe
xl = pd.read_excel('hd25.xlsx')

#%%
#Accessing Necessary Sheets
hp = pd.read_excel('hd25.xlsx',sheet_name = 'house prices')
hp = hp.drop(hp.index[range(5)])
earn = pd.read_excel('hd25.xlsx',sheet_name = 'retail prices and earnings')

#%%
#calculating average house prices
ahp = []
aearn = []
for k in range(0,43):
    ahp.append((hp.iloc[4*k,1] + hp.iloc[4*k+1,1] + hp.iloc[4*k+2,1] + hp.iloc[4*k+3,1])/4)
    aearn.append(earn.iloc[k+1,2])

#Comparing Earnings/Price
x = [2000,6000,10000,14000,18000,22000,26000]
y = [10000,50000,90000,130000,170000,210000]

plt.xticks(x)
plt.yticks(y)
plt.xlabel('Average Annual Nominal Earnings')
plt.ylabel('Average UK House Price')
plt.title('Comparison of average earnings and house prices over time')
plt.plot(aearn,ahp)
plt.show()

#%%
#Time to buy
time = []
for l in range(0,43):
    time.append(ahp[l]/aearn[l])

x = [1974,1980,1986,1992,1998,2004,2010,2016]
y = [4,5,6,7,8]

plt.xticks(x)
plt.yticks(y)
plt.xlabel('Year')
plt.ylabel('Relative House Price')
plt.title('Relative Housing Cost Over Time')
plt.plot(earn.iloc[1:45,0],time)
plt.show()

#%%
#Direct Comparison

x = [1974,1980,1986,1992,1998,2004,2010,2016]
y = [0,50000,100000,150000,200000]

plt.xticks(x)
plt.yticks(y)
plt.xlabel('Year')
plt.ylabel('Average £')
plt.title('Average Earnings/House Price Over Time')
plt.plot(earn.iloc[1:45,0],ahp , label = 'House Price')
plt.plot(earn.iloc[1:45,0],aearn, label = 'Earnings')
plt.legend(loc = "center right", bbox_to_anchor = (1.4,0.5))
plt.show()
    