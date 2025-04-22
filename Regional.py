# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 10:42:23 2025

@author: FiercePC
"""
#PART B :- HOME OWNERSHIP DATA ANALYSIS
#Region and Country Analysis

#Importing Libraries
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

#Excel file reading and manipulating into dataframe
xl = pd.read_excel('hd25.xlsx')

#%%
#Accessing sheet related to this
Type = pd.read_excel('hd25.xlsx',sheet_name = 'Type by Region')
del Type['Unnamed: 6']
Type = Type.rename(columns={'Housing tenure by region, 1996 to 2016':'Region','Unnamed: 1':'1996','Unnamed: 2':'2001','Unnamed: 3':'2006','Unnamed: 4':'2011','Unnamed: 5':'2016','Unnamed: 7':'1996','Unnamed: 8':'2001','Unnamed: 9':'2006','Unnamed: 10':'2011','Unnamed: 11':'2016'})

#%%
#Analysis By Region
reg = Type.iloc[3:12,0:11]

x = [1996,2001,2006,2011,2016]
y = [50,55,60,65,70,75]

plt.xticks(x) 
plt.yticks(y) 
plt.xlabel('Year')
plt.ylabel('Home Ownership(%)')
plt.title('Percentage Home Ownership By Region [1996-2016]')
for i in range(0,9):
    plt.plot(x,(reg.iloc[i,6:11])*100, label = reg.iloc[i,0])

plt.legend(loc = "center right", bbox_to_anchor = (1.4,0.5))
plt.show()

#%%
#Analysis By Country
ctr = Type.iloc[13:17,0:11]

x = [1996,2001,2006,2011,2016] 
y = [58,62,66,70,74]

plt.xticks(x)
plt.yticks(y)
plt.xlabel('Year')
plt.ylabel('Home Ownership(%)')
plt.title("Home Ownership By Country [1996-2016]")
for j in range(0,4):
    plt.plot(x,(ctr.iloc[j,6:11])*100, label = ctr.iloc[j,0])

plt.legend(loc = "center right", bbox_to_anchor = (1.4,0.5))
plt.show()
