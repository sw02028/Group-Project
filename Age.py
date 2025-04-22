# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 10:56:35 2025

@author: FiercePC
"""
#PART B :- HOME OWNERSHIP DATA ANALYSIS
#Age Analysis

#Importing Libraries
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

#Excel file reading and manipulating into dataframe
xl = pd.read_excel('hd25.xlsx')

#%%
#Accessing Required Sheet and editing
age = pd.read_excel('hd25.xlsx',sheet_name = 'Age group')
age = age.drop(age.index[range(3)])
del age['Unnamed: 6']

#%%
#Small Ranges
x = [1996,2001,2006,2011,2016]
y = [10,20,30,40,50,60,70,80]

plt.xticks(x) 
plt.yticks(y) 
plt.xlabel('Year')
plt.ylabel('Home Ownership(%)')
plt.title('Percentage Home Ownership By Age [1996-2016]')
for i in range(0,7):
    plt.plot(x,(age.iloc[i,6:11])*100, label = age.iloc[i,0])

plt.legend(loc = "center right", bbox_to_anchor = (1.4,0.5))
plt.show()

#%%
#Large Ranges
x = [1996,2001,2006,2011,2016]
y = [30,40,50,60,70,80]

plt.xticks(x) 
plt.yticks(y) 
plt.xlabel('Year')
plt.ylabel('Home Ownership(%)')
plt.title('Percentage Home Ownership By Age [1996-2016]')
for j in range(0,3):
    plt.plot(x,(age.iloc[j+8,6:11])*100, label = age.iloc[j+8,0])
plt.plot(x,age.iloc[12,6:11]*100, label = 'Total Population')

plt.legend(loc = "center right", bbox_to_anchor = (1.4,0.5))
plt.show()