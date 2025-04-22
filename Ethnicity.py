# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 11:20:18 2025

@author: FiercePC
"""
#PART B :- HOME OWNERSHIP DATA ANALYSIS
#Ethnicity Analysis

#Importing Libraries
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

#Excel file reading and manipulating into dataframe
xl = pd.read_excel('hd25.xlsx')

#%%
#Accessing Necessary Sheets
eth = pd.read_excel('hd25.xlsx',sheet_name = 'Ethnicity')
eth = eth.drop(eth.index[range(3)])
del eth['Unnamed: 5']
birth = pd.read_excel('hd25.xlsx',sheet_name = 'Country of birth')
birth = birth.drop(birth.index[range(3)])
del birth['Unnamed: 6']

#%%
#Ethnicity
x = [2001,2006,2011,2016]
y = [30,40,50,60,70]

plt.xticks(x)
plt.yticks(y)
plt.xlabel('Year')
plt.ylabel('Home Ownership(%)')
plt.title('Percentage Home Ownership By Ethnicity [2001-2016]')
for i in range(0,5):
    plt.plot(x,(eth.iloc[i,5:9])*100, label = eth.iloc[i,0])

plt.legend(loc = "center right", bbox_to_anchor = (1.5,0.5))
plt.show()

#%%
#UK Born?
x = [1996,2001,2006,2011,2016]
y = [45,50,55,60,65,70,75]

plt.xticks(x)
plt.yticks(y)
plt.xlabel('Year')
plt.ylabel('Home Ownership(%)')
plt.title('Percentage Home Ownership By Birthplace [1996-2016]')
for j in range(0,3):
    plt.plot(x,(birth.iloc[j,6:11])*100, label = birth.iloc[j,0])

plt.legend(loc = "center right", bbox_to_anchor = (1.31,0.5))
plt.show()