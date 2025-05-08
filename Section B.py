# -*- coding: utf-8 -*-
"""
Created on Wed May  7 10:34:26 2025

@author: FiercePC
"""
###Part B: Data Analysis and Visualisation

#Importing Libraries
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#Excel file reading and manipulating into dataframe
xl = pd.read_excel('Housing data 2025.xlsx')

###Prescribed Analysis
#####HOUSING ANALYSIS

### Housing Analysis Over Different Regions
#%%
#Accessing sheet related to this
Type = pd.read_excel('Housing data 2025.xlsx',sheet_name = 'Type by Region')
#Removing unneeded data lines
del Type['Unnamed: 6']
Type = Type.rename(columns={'Housing tenure by region, 1996 to 2016':'Region','Unnamed: 1':'1996','Unnamed: 2':'2001','Unnamed: 3':'2006','Unnamed: 4':'2011','Unnamed: 5':'2016','Unnamed: 7':'1996','Unnamed: 8':'2001','Unnamed: 9':'2006','Unnamed: 10':'2011','Unnamed: 11':'2016'})

###Analysis By Region
#Restricting Sheet columns and rows
reg = Type.iloc[3:12,0:11]
#Setting x,y ticks
x = [1996,2001,2006,2011,2016]
y = [50,55,60,65,70,75]
plt.xticks(x) 
plt.yticks(y)
#Labelling Graph 
plt.xlabel('Year')
plt.ylabel('Home Ownership(%)')
plt.title('Percentage Home Ownership By Region [1996-2016]')
#Producing plot
for i in range(0,9):
    plt.plot(x,(reg.iloc[i,6:11])*100, label = reg.iloc[i,0])
plt.legend(loc = "center right", bbox_to_anchor = (1.4,0.5))
plt.show()

###Analysis By Country
#Restricting Sheet columns and rows
ctr = Type.iloc[13:17,0:11]
#Setting x,y ticks
x = [1996,2001,2006,2011,2016] 
y = [58,62,66,70,74]
plt.xticks(x)
plt.yticks(y)
#Labelling Graph
plt.xlabel('Year')
plt.ylabel('Home Ownership(%)')
plt.title("Home Ownership By Country [1996-2016]")
#Producing plot
for j in range(0,4):
    plt.plot(x,(ctr.iloc[j,6:11])*100, label = ctr.iloc[j,0])
plt.legend(loc = "center right", bbox_to_anchor = (1.4,0.5))
plt.show()

### Affordability over time
#%%
#Accessing Necessary Sheets
hp = pd.read_excel('Housing data 2025.xlsx',sheet_name = 'house prices')
hp = hp.drop(hp.index[range(5)])
earn = pd.read_excel('Housing data 2025.xlsx',sheet_name = 'retail prices and earnings')
#calculating average house prices
ahp = []
aearn = []
for k in range(0,43):
    ahp.append((hp.iloc[4*k,1] + hp.iloc[4*k+1,1] + hp.iloc[4*k+2,1] + hp.iloc[4*k+3,1])/4)
    aearn.append(earn.iloc[k+1,2])

#Comparing Earnings/Price
#Setting x,y ticks
x = [2000,6000,10000,14000,18000,22000,26000]
y = [10000,50000,90000,130000,170000,210000]
plt.xticks(x)
plt.yticks(y)
#Labelling Graph
plt.xlabel('Average Annual Nominal Earnings')
plt.ylabel('Average UK House Price')
plt.title('Comparison of average earnings and house prices over time')
#Producing plot
plt.plot(aearn,ahp)
plt.show()

#Time to buy
#Calculating time variable
time = []
for l in range(0,43):
    time.append(ahp[l]/aearn[l])
#Setting x,y ticks
x = [1974,1980,1986,1992,1998,2004,2010,2016]
y = [4,5,6,7,8]
plt.xticks(x)
plt.yticks(y)
#Labelling Graph
plt.xlabel('Year')
plt.ylabel('Relative House Price')
plt.title('Relative Housing Cost Over Time')
#Producing Plot
plt.plot(earn.iloc[1:45,0],time)
plt.show()

#Direct Comparison
#setting x,y ticks
x = [1974,1980,1986,1992,1998,2004,2010,2016]
y = [0,50000,100000,150000,200000]
plt.xticks(x)
plt.yticks(y)
#labelling graph
plt.xlabel('Year')
plt.ylabel('Average £')
plt.title('Average Earnings/House Price Over Time')
#producing plots
plt.plot(earn.iloc[1:45,0],ahp , label = 'House Price')
plt.plot(earn.iloc[1:45,0],aearn, label = 'Earnings')
plt.legend(loc = "center right", bbox_to_anchor = (1.4,0.5))
plt.show()

### Housing Age Analysis
#%%
#Accessing Required Sheet and editing
age = pd.read_excel('Housing data 2025.xlsx',sheet_name = 'Age group')
age = age.drop(age.index[range(3)])
del age['Unnamed: 6']

##Housing Analysis
#Small Ranges
#Setting x,y ticks
x = [1996,2001,2006,2011,2016]
y = [10,20,30,40,50,60,70,80]
plt.xticks(x) 
plt.yticks(y) 
#Labelling graph
plt.xlabel('Year')
plt.ylabel('Home Ownership(%)')
plt.title('Percentage Home Ownership By Age [1996-2016]')
#producing plots
for i in range(0,7):
    plt.plot(x,(age.iloc[i,6:11])*100, label = age.iloc[i,0])
plt.legend(loc = "center right", bbox_to_anchor = (1.4,0.5))
plt.show()

#Large Ranges
#Setting x,y ticks
x = [1996,2001,2006,2011,2016]
y = [30,40,50,60,70,80]
plt.xticks(x) 
plt.yticks(y) 
#Labelling Graph
plt.xlabel('Year')
plt.ylabel('Home Ownership(%)')
plt.title('Percentage Home Ownership By Age [1996-2016]')
#Producing plots
for j in range(0,3):
    plt.plot(x,(age.iloc[j+8,6:11])*100, label = age.iloc[j+8,0])
plt.plot(x,age.iloc[12,6:11]*100, label = 'Total Population')
plt.legend(loc = "center right", bbox_to_anchor = (1.4,0.5))
plt.show()

###Housing Ethnicity Analysis
#%%
#Accessing Necessary Sheets
eth = pd.read_excel('Housing data 2025.xlsx',sheet_name = 'Ethnicity')
eth = eth.drop(eth.index[range(3)])
del eth['Unnamed: 5']
birth = pd.read_excel('Housing data 2025.xlsx',sheet_name = 'Country of birth')
birth = birth.drop(birth.index[range(3)])
del birth['Unnamed: 6']

##Housing Analysis
#Ethnicity
#setting x,y ticks
x = [2001,2006,2011,2016]
y = [30,40,50,60,70]
plt.xticks(x)
plt.yticks(y)
#labelling graph
plt.xlabel('Year')
plt.ylabel('Home Ownership(%)')
plt.title('Percentage Home Ownership By Ethnicity [2001-2016]')
#Producing plot
for i in range(0,5):
    plt.plot(x,(eth.iloc[i,5:9])*100, label = eth.iloc[i,0])
plt.legend(loc = "center right", bbox_to_anchor = (1.5,0.5))
plt.show()

#UK Born?
#setting x,y ticks
x = [1996,2001,2006,2011,2016]
y = [45,50,55,60,65,70,75]
plt.xticks(x)
plt.yticks(y)
#labelling graphs
plt.xlabel('Year')
plt.ylabel('Home Ownership(%)')
plt.title('Percentage Home Ownership By Birthplace [1996-2016]')
#producing plots
for j in range(0,3):
    plt.plot(x,(birth.iloc[j,6:11])*100, label = birth.iloc[j,0])
plt.legend(loc = "center right", bbox_to_anchor = (1.31,0.5))
plt.show()

#####RENTING ANALYSIS
'''REGIONS + COUNTRIES'''
df1 = pd.read_excel('Housing data 2025.xlsx', sheet_name = 'Type by Region', header=None)

# remove trailing columns
df1 = df1.iloc[0:,0:12]

# remove null column
df1 = df1.drop(6, axis = 1)

# rename columns
df1.columns = ['Region', '1996', '2001', '2006', '2011', '2016', '1996', '2001', '2006', '2011', '2016']


''' PRIVATE RENTING '''
# extract all the relevant rows for private renting
priv_areas = df1.iloc[22:31,0:]
priv_countries = df1.iloc[32:36,0:]
priv_reg_total = df1.iloc[37:38,0:]

# join the previous dataframes together
priv_by_regions = pd.concat([priv_areas, priv_countries, priv_reg_total], ignore_index=True)


#%% 

''' PRIVATE GRAPHS '''
priv_dfs = [priv_areas, priv_countries, priv_reg_total]
percentage_priv = []

# loop through each sector of the data and separate the percentage portion of it
for dfs in priv_dfs:
    region_name = dfs.iloc[0:, 0]
    percentages = dfs.iloc[0:, -5:]
    new_df = pd.concat([region_name, percentages], axis = 1)
    percentage_priv.append(new_df)
    
# regions, numbers

plt.figure(figsize=(10,5))

# loop over each row 
for i in priv_areas.index:
    row = priv_areas.loc[i]
    
    # x gets the column names (the years)
    x = priv_areas.columns[1:6]
    # y gets the data from each row
    y = row[1:6].values

    # plot the values of each row
    plt.plot(x, y, label=row['Region'], marker='o')

plt.xlabel('Year')
plt.ylabel('Number of People')
plt.title('Number of People in Private Renting by Region [1996-2016]')
plt.legend()
plt.show()

# regions, percentages

priv_areas_percent = percentage_priv[0]
plt.figure(figsize=(10,5))

# loop over each row 
for i in priv_areas_percent.index:
    row = priv_areas_percent.loc[i]
    
    # x gets the column names (the years)
    x = priv_areas_percent.columns[1:6]
    # y gets the data from each row, multiplied by 100 for percentage
    y = row[1:6].values * 100

    # plot the values of each row
    plt.plot(x, y, label=row['Region'], marker='o')

plt.xlabel('Year')
plt.ylabel('Private Renting (%)')
plt.title('Percentage of People in Private Renting by Region [1996-2016]')
plt.legend()
plt.show()

# countries, numbers

plt.figure(figsize=(10,5))

# loop over each row 
for i in priv_countries.index:
    row = priv_countries.loc[i]
    
    # x gets the column names (the years)
    x = priv_countries.columns[1:6]
    # y gets the data from each row
    y = row[1:6].values

    # plot the values of each row
    plt.plot(x, y, label=row['Region'], marker='o')

plt.xlabel('Year')
plt.ylabel('Number of People')
plt.title('Number of People in Private Renting by Country [1996-2016]')
plt.legend()
plt.show()

# countries, percentages

priv_countries_percent = percentage_priv[1]
plt.figure(figsize=(10,5))

# loop over each row 
for i in priv_countries_percent.index:
    row = priv_countries_percent.loc[i]
    
    # x gets the column names (the years)
    x = priv_countries_percent.columns[1:6]
    # y gets the data from each row, multiplied by 100 for percentage
    y = row[1:6].values * 100

    # plot the values of each row
    plt.plot(x, y, label=row['Region'], marker='o')

plt.xlabel('Year')
plt.ylabel('Private Renting (%)')
plt.title('Percentage of People in Private Renting by Country [1996-2016]')
plt.legend()
plt.show()

# uk, numbers

plt.figure(figsize=(10,5))

# x is the column names a.k.a. the years
x = priv_reg_total.columns[1:6]
# y is the data
y = priv_reg_total.iloc[0, 1:6]

# plot the values of each row
plt.plot(x, y, marker='o')

plt.xlabel('Year')
plt.ylabel('Number of People')
plt.title('Number of People in Private Renting in the UK [1996-2016]')
plt.show()

# uk, percentages

priv_reg_total_percent = percentage_priv[2]
plt.figure(figsize=(10,5))

# x is the column names a.k.a. the years
x = priv_reg_total_percent.columns[1:6]
# y is the data, multiply by 100 to get percentage
y = priv_reg_total_percent.iloc[0, 1:6] * 100

# plot the values of each row
plt.plot(x, y, marker='o')

plt.xlabel('Year')
plt.ylabel('Private Renting (%)')
plt.title('Percentage of People in Private Renting in the UK [1996-2016]')
plt.show()

#%%

''' SOCIAL RENTING '''
# extract all the relevant rows for social renting
soc_areas = df1.iloc[40:49,0:]
soc_countries = df1.iloc[50:54,0:]
soc_reg_total = df1.iloc[55:56,0:]

# join the previous dataframes together
soc_by_regions = pd.concat([soc_areas, soc_countries, soc_reg_total], ignore_index=True)

total_renting_regions = priv_reg_total.iloc[0, 1:] + soc_reg_total.iloc[0, 1:]
total_renting_regions_num = pd.DataFrame(total_renting_regions.iloc[0:5])

# uk, numbers

plt.figure(figsize=(10,5))

# x is the column names a.k.a. the years
x = total_renting_regions_num.index
# y is the data
y = total_renting_regions_num.values

# plot the values of each row
plt.plot(x, y, marker='o')

plt.xlabel('Year')
plt.ylabel('Number of People')
plt.title('Number of People in Renting in the UK [1996-2016]')
plt.show()

# uk, percentages
total_renting_regions_perc = pd.DataFrame(total_renting_regions.iloc[5:])
plt.figure(figsize=(10,5))

# x is the column names a.k.a. the years
x = total_renting_regions_perc.index

# y is the data, multiply by 100 to get percentage
y = total_renting_regions_perc.values * 100

# plot the values of each row
plt.plot(x, y, marker='o')

plt.xlabel('Year')
plt.ylabel('Private Renting (%)')
plt.title('Percentage of People in Renting in the UK [1996-2016]')
plt.show()


#print(total_renting_regions)
#print(soc_by_regions)

#%% 

''' SOCIAL GRAPHS '''
soc_dfs = [soc_areas, soc_countries, soc_reg_total]
percentage_soc = []

# loop through each sector of the data and separate the percentage portion of it
for dfs in soc_dfs:
    region_name = dfs.iloc[0:, 0]
    percentages = dfs.iloc[0:, -5:]
    new_df = pd.concat([region_name, percentages], axis = 1)
    percentage_soc.append(new_df)
    
# regions, numbers

plt.figure(figsize=(10,5))

# loop over each row 
for i in soc_areas.index:
    row = soc_areas.loc[i]
    
    # x gets the column names (the years)
    x = soc_areas.columns[1:6]
    # y gets the data from each row
    y = row[1:6].values

    # plot the values of each row
    plt.plot(x, y, label=row['Region'], marker='o')

plt.xlabel('Year')
plt.ylabel('Number of People')
plt.title('Number of People in Social Renting by Region [1996-2016]')
plt.legend()
plt.show()

# regions, percentages

soc_areas_percent = percentage_soc[0]
plt.figure(figsize=(10,5))

# loop over each row 
for i in soc_areas_percent.index:
    row = soc_areas_percent.loc[i]
    
    # x gets the column names (the years)
    x = soc_areas_percent.columns[1:6]
    # y gets the data from each row, multiplied by 100 for percentage
    y = row[1:6].values * 100

    # plot the values of each row
    plt.plot(x, y, label=row['Region'], marker='o')

plt.xlabel('Year')
plt.ylabel('Social Renting (%)')
plt.title('Percentage of People in Social Renting by Region [1996-2016]')
plt.legend()
plt.show()

# countries, numbers

plt.figure(figsize=(10,5))

# loop over each row 
for i in soc_countries.index:
    row = soc_countries.loc[i]
    
    # x gets the column names (the years)
    x = soc_countries.columns[1:6]
    # y gets the data from each row
    y = row[1:6].values

    # plot the values of each row
    plt.plot(x, y, label=row['Region'], marker='o')

plt.xlabel('Year')
plt.ylabel('Number of People')
plt.title('Number of People in Social Renting by Country [1996-2016]')
plt.legend()
plt.show()

# countries, percentages

soc_countries_percent = percentage_soc[1]
plt.figure(figsize=(10,5))

# loop over each row 
for i in soc_countries_percent.index:
    row = soc_countries_percent.loc[i]
    
    # x gets the column names (the years)
    x = soc_countries_percent.columns[1:6]
    # y gets the data from each row, multiplied by 100 for percentage
    y = row[1:6].values * 100

    # plot the values of each row
    plt.plot(x, y, label=row['Region'], marker='o')

plt.xlabel('Year')
plt.ylabel('Social Renting (%)')
plt.title('Percentage of People in Social Renting by Country [1996-2016]')
plt.legend()
plt.show()

# uk, numbers

plt.figure(figsize=(10,5))

# x is the column names a.k.a. the years
x = soc_reg_total.columns[1:6]
# y is the data
y = soc_reg_total.iloc[0, 1:6]

# plot the values of each row
plt.plot(x, y, marker='o')

plt.xlabel('Year')
plt.ylabel('Number of People')
plt.title('Number of People in Social Renting in the UK [1996-2016]')
plt.show()

# uk, percentages

soc_reg_total_percent = percentage_soc[2]
plt.figure(figsize=(10,5))

# x is the column names a.k.a. the years
x = soc_reg_total_percent.columns[1:6]
# y is the data, multiply by 100 to get percentage
y = soc_reg_total_percent.iloc[0, 1:6] * 100

# plot the values of each row
plt.plot(x, y, marker='o')

plt.xlabel('Year')
plt.ylabel('Social Renting (%)')
plt.title('Percentage of People in Social Renting in the UK [1996-2016]')
plt.show()

#%% 

''' AGE GROUPS '''
df2 = pd.read_excel('Housing data 2025.xlsx', sheet_name = 'Age group', header=None)

# remove trailing columns
df2 = df2.iloc[0:,0:12]

# remove null column
df2 = df2.drop(6, axis = 1)

# rename columns
df2.columns = ['Ages', '1996', '2001', '2006', '2011', '2016', '1996', '2001', '2006', '2011', '2016']

''' PRIVATE RENTING '''
# extract all the relevant rows for private renting
priv_specific = df2.iloc[19:26,0:12]
priv_general = df2.iloc[27:30,0:12]
priv_age_total = df2.iloc[31:32,0:12]

# join the previous dataframes together
priv_by_age = pd.concat([priv_specific, priv_general, priv_age_total], ignore_index=True)

#print(priv_by_age)

#%% 

''' PRIVATE GRAPHS '''
priv_dfs = [priv_specific, priv_general]
percentage_priv = []

# loop through each sector of the data and separate the percentage portion of it
for dfs in priv_dfs:
    age_group = dfs.iloc[0:, 0]
    percentages = dfs.iloc[0:, -5:]
    new_df = pd.concat([age_group, percentages], axis = 1)
    percentage_priv.append(new_df)
    
# specific, numbers

plt.figure(figsize=(10,5))

# loop over each row 
for i in priv_specific.index:
    row = priv_specific.loc[i]
    
    # x gets the column names (the years)
    x = priv_specific.columns[1:6]
    # y gets the data from each row
    y = row[1:6].values

    # plot the values of each row
    plt.plot(x, y, label=row['Ages'], marker='o')

plt.xlabel('Year')
plt.ylabel('Number of People')
plt.title('Number of People in Private Renting by Age Group (9 Year Gaps) [1996-2016]')
plt.legend()
plt.show()

# specific, percentages

priv_specific_percent = percentage_priv[0]
plt.figure(figsize=(10,5))

# loop over each row 
for i in priv_specific_percent.index:
    row = priv_specific_percent.loc[i]
    
    # x gets the column names (the years)
    x = priv_specific_percent.columns[1:6]
    # y gets the data from each row, multiplied by 100 for percentage
    y = row[1:6].values * 100

    # plot the values of each row
    plt.plot(x, y, label=row['Ages'], marker='o')

plt.xlabel('Year')
plt.ylabel('Private Renting (%)')
plt.title('Percentage of People in Private Renting by Age Group (9 Year Gaps) [1996-2016]')
plt.legend()
plt.show()

# general, numbers

plt.figure(figsize=(10,5))

# loop over each row 
for i in priv_general.index:
    row = priv_general.loc[i]
    
    # x gets the column names (the years)
    x = priv_general.columns[1:6]
    # y gets the data from each row
    y = row[1:6].values

    # plot the values of each row
    plt.plot(x, y, label=row['Ages'], marker='o')

plt.xlabel('Year')
plt.ylabel('Number of People')
plt.title('Number of People in Private Renting by Age Group (15 Year Gaps) [1996-2016]')
plt.legend()
plt.show()

# general, percentages

priv_general_percent = percentage_priv[1]
plt.figure(figsize=(10,5))

# loop over each row 
for i in priv_general_percent.index:
    row = priv_general_percent.loc[i]
    
    # x gets the column names (the years)
    x = priv_general_percent.columns[1:6]
    # y gets the data from each row, multiplied by 100 for percentage
    y = row[1:6].values * 100

    # plot the values of each row
    plt.plot(x, y, label=row['Ages'], marker='o')

plt.xlabel('Year')
plt.ylabel('Private Renting (%)')
plt.title('Percentage of People in Private Renting by Age Group (15 Year Gaps) [1996-2016]')
plt.legend()
plt.show()

#%%

''' SOCIAL RENTING '''
# extract all the relevant rows for social renting
soc_specific = df2.iloc[34:41,0:12]
soc_general = df2.iloc[42:45,0:12]
soc_age_total = df2.iloc[46:47,0:12]

# join the previous dataframes together
soc_by_age = pd.concat([soc_specific, soc_general, soc_age_total], ignore_index=True)

#print(soc_by_age)

#%% 

''' SOCIAL GRAPHS '''
soc_dfs = [soc_specific, soc_general]
percentage_soc = []

# loop through each sector of the data and separate the percentage portion of it
for dfs in soc_dfs:
    age_group = dfs.iloc[0:, 0]
    percentages = dfs.iloc[0:, -5:]
    new_df = pd.concat([age_group, percentages], axis = 1)
    percentage_soc.append(new_df)
    
# specific, numbers

plt.figure(figsize=(10,5))

# loop over each row 
for i in soc_specific.index:
    row = soc_specific.loc[i]
    
    # x gets the column names (the years)
    x = soc_specific.columns[1:6]
    # y gets the data from each row
    y = row[1:6].values

    # plot the values of each row
    plt.plot(x, y, label=row['Ages'], marker='o')

plt.xlabel('Year')
plt.ylabel('Number of People')
plt.title('Number of People in Social Renting by by Age Group (9 Year Gaps) [1996-2016]')
plt.legend()
plt.show()

# specific, percentages

soc_specific_percent = percentage_soc[0]
plt.figure(figsize=(10,5))

# loop over each row 
for i in soc_specific_percent.index:
    row = soc_specific_percent.loc[i]
    
    # x gets the column names (the years)
    x = soc_specific_percent.columns[1:6]
    # y gets the data from each row, multiplied by 100 for percentage
    y = row[1:6].values * 100

    # plot the values of each row
    plt.plot(x, y, label=row['Ages'], marker='o')

plt.xlabel('Year')
plt.ylabel('Social Renting (%)')
plt.title('Percentage of People in Social Renting by by Age Group (9 Year Gaps) [1996-2016]')
plt.legend()
plt.show()

# general, numbers

plt.figure(figsize=(10,5))

# loop over each row 
for i in soc_general.index:
    row = soc_general.loc[i]
    
    # x gets the column names (the years)
    x = soc_general.columns[1:6]
    # y gets the data from each row
    y = row[1:6].values

    # plot the values of each row
    plt.plot(x, y, label=row['Ages'], marker='o')

plt.xlabel('Year')
plt.ylabel('Number of People')
plt.title('Number of People in Social Renting by Age Group (15 Year Gaps) [1996-2016]')
plt.legend()
plt.show()

# general, percentages

soc_general_percent = percentage_soc[1]
plt.figure(figsize=(10,5))

# loop over each row 
for i in soc_general_percent.index:
    row = soc_general_percent.loc[i]
    
    # x gets the column names (the years)
    x = soc_general_percent.columns[1:6]
    # y gets the data from each row, multiplied by 100 for percentage
    y = row[1:6].values * 100

    # plot the values of each row
    plt.plot(x, y, label=row['Ages'], marker='o')

plt.xlabel('Year')
plt.ylabel('Social Renting (%)')
plt.title('Percentage of People in Social Renting by Age Group (15 Year Gaps) [1996-2016]')
plt.legend()
plt.show()

#%%

''' ETHNICITY '''
df3 = pd.read_excel('Housing data 2025.xlsx', sheet_name = 'Ethnicity', header=None)

# remove trailing columns
df3 = df3.iloc[0:,0:10]

# remove null column
df3 = df3.drop(5, axis = 1)

# rename columns
df3.columns = ['Ethnicity', '2001', '2006', '2011', '2016', '2001', '2006', '2011', '2016']

''' PRIVATE RENTING '''
# extract all the relevant rows for private renting
priv_eth = df3.iloc[13:18,0:10]
priv_eth_total = df3.iloc[19:20,0:10]

# join the previous dataframes together
priv_by_ethnicity = pd.concat([priv_eth, priv_eth_total], ignore_index=True)

#print(priv_by_ethnicity)

#%%

''' PRIVATE GRAPHS '''
# separate the percentage portion of the data
ethnicity = priv_eth.iloc[0:, 0]
percentages = priv_eth.iloc[0:, -4:]
priv_eth_percent = pd.concat([ethnicity, percentages], axis = 1)
    
# ethnicity, numbers

plt.figure(figsize=(10,5))

# loop over each row 
for i in priv_eth.index:
    row = priv_eth.loc[i]
    
    # x gets the column names (the years)
    x = priv_eth.columns[1:5]
    # y gets the data from each row
    y = row[1:5].values

    # plot the values of each row
    plt.plot(x, y, label=row['Ethnicity'], marker='o')

plt.xlabel('Year')
plt.ylabel('Number of People')
plt.title('Number of People in Private Renting by Ethnicity [2001-2016]')
plt.legend()
plt.show()

# ethnicity, percentages

plt.figure(figsize=(10,5))

# loop over each row 
for i in priv_eth_percent.index:
    row = priv_eth_percent.loc[i]
    
    # x gets the column names (the years)
    x = priv_eth_percent.columns[1:5]
    # y gets the data from each row, multiplied by 100 for percentage
    y = row[1:5].values * 100

    # plot the values of each row
    plt.plot(x, y, label=row['Ethnicity'], marker='o')

plt.xlabel('Year')
plt.ylabel('Private Renting (%)')
plt.title('Percentage of People in Private Renting by Ethnicity [2001-2016]')
plt.legend()
plt.show()

#%%

''' SOCIAL RENTING '''
# extract all the relevant rows for social renting
soc_eth = df3.iloc[22:27,0:10]
soc_eth_total = df3.iloc[28:29,0:10]

# join the previous dataframes together
soc_by_ethnicity = pd.concat([soc_eth, soc_eth_total], ignore_index=True)

#print(soc_by_ethnicity)

#%% 

''' SOCIAL GRAPHS '''
# separate the percentage portion of the data
ethnicity = soc_eth.iloc[0:, 0]
percentages = soc_eth.iloc[0:, -4:]
soc_eth_percent = pd.concat([ethnicity, percentages], axis = 1)

# ethnicity, numbers

plt.figure(figsize=(10,5))

# loop over each row 
for i in soc_eth.index:
    row = soc_eth.loc[i]
    
    # x gets the column names (the years)
    x = soc_eth.columns[1:5]
    # y gets the data from each row
    y = row[1:5].values

    # plot the values of each row
    plt.plot(x, y, label=row['Ethnicity'], marker='o')

plt.xlabel('Year')
plt.ylabel('Number of People')
plt.title('Number of People in Social Renting by Ethnicity [2001-2016]')
plt.legend()
plt.show()

# ethnicity, percentages

plt.figure(figsize=(10,5))

# loop over each row 
for i in soc_eth_percent.index:
    row = soc_eth_percent.loc[i]
    
    # x gets the column names (the years)
    x = soc_eth_percent.columns[1:5]
    # y gets the data from each row, multiplied by 100 for percentage
    y = row[1:5].values * 100

    # plot the values of each row
    plt.plot(x, y, label=row['Ethnicity'], marker='o')

plt.xlabel('Year')
plt.ylabel('Social Renting (%)')
plt.title('Percentage of People in Social Renting by Ethnicity [2001-2016]')
plt.legend()
plt.show()

#%% 

''' COUNTRY OF BIRTH '''
df4 = pd.read_excel('Housing data 2025.xlsx', sheet_name = 'Country of birth', header=None)

# remove trailing columns
df4 = df4.iloc[0:,0:12]

# remove null column
df4 = df4.drop(6, axis = 1)

# rename columns
df4.columns = ['Country', '1996', '2001', '2006', '2011', '2016', '1996', '2001', '2006', '2011', '2016']

''' PRIVATE RENTING '''
# extract all the relevant rows for private renting
priv_born = df4.iloc[9:11,0:12]
                    
priv_by_country_born = pd.concat([priv_born, df4.iloc[11:12,0:12]], ignore_index=True) 

#print(priv_by_country_born)

#%%

''' PRIVATE GRAPHS '''
# separate the percentage portion of the data
country = priv_born.iloc[0:, 0]
percentages = priv_born.iloc[0:, -5:]
priv_born_percent = pd.concat([country, percentages], axis = 1)

# country born, numbers

plt.figure(figsize=(10,5))

# loop over each row 
for i in priv_born.index:
    row = priv_born.loc[i]
    
    # x gets the column names (the years)
    x = priv_born.columns[1:6]
    # y gets the data from each row
    y = row[1:6].values

    # plot the values of each row
    plt.plot(x, y, label=row['Country'], marker='o')

plt.xlabel('Year')
plt.ylabel('Number of People')
plt.title('Number of People in Private Renting by Country Born [1996-2016]')
plt.legend()
plt.show()

# country born, percentages

plt.figure(figsize=(10,5))

# loop over each row 
for i in priv_born_percent.index:
    row = priv_born_percent.loc[i]
    
    # x gets the column names (the years)
    x = priv_born_percent.columns[1:6]
    # y gets the data from each row, multiplied by 100 for percentage
    y = row[1:6].values * 100

    # plot the values of each row
    plt.plot(x, y, label=row['Country'], marker='o')

plt.xlabel('Year')
plt.ylabel('Private Renting (%)')
plt.title('Percentage of People in Private Renting by Country Born [1996-2016]')
plt.legend()
plt.show()

#%%

''' SOCIAL RENTING '''
soc_born = df4.iloc[14:16,0:12]
soc_by_country_born = pd.concat([soc_born, df4.iloc[16:17,0:12]], ignore_index=True) 

#print(soc_by_country_born)

#%%

''' SOCIAL GRAPHS '''
# separate the percentage portion of the data
country = soc_born.iloc[0:, 0]
percentages = soc_born.iloc[0:, -5:]
soc_born_percent = pd.concat([country, percentages], axis = 1)

# country born, numbers

plt.figure(figsize=(10,5))

# loop over each row 
for i in soc_born.index:
    row = soc_born.loc[i]
    
    # x gets the column names (the years)
    x = soc_born.columns[1:6]
    # y gets the data from each row
    y = row[1:6].values

    # plot the values of each row
    plt.plot(x, y, label=row['Country'], marker='o')

plt.xlabel('Year')
plt.ylabel('Number of People')
plt.title('Number of People in Social Renting by Country Born [1996-2016]')
plt.legend()
plt.show()

# country born, percentages

plt.figure(figsize=(10,5))

# loop over each row 
for i in soc_born_percent.index:
    row = soc_born_percent.loc[i]
    
    # x gets the column names (the years)
    x = soc_born_percent.columns[1:6]
    # y gets the data from each row, multiplied by 100 for percentage
    y = row[1:6].values * 100

    # plot the values of each row
    plt.plot(x, y, label=row['Country'], marker='o')

plt.xlabel('Year')
plt.ylabel('Social Renting (%)')
plt.title('Percentage of People in Social Renting by Country Born [1996-2016]')
plt.legend()
plt.show()

###6: Regression Lines
#%%
#Ethnicity Housing Regression lines
reg = eth.iloc[3:12,0:11]
x = np.array([2001, 2006, 2011, 2016])
y = np.array([30,40,50,60,70])
#creating regression model
lin_reg = LinearRegression(fit_intercept = True)
x_rs = x.reshape(-1,1)
#creating ticks and labels
plt.xticks(x)
plt.yticks(y)
plt.xlabel('Year')
plt.ylabel('Home Ownership(%)')
plt.title('Percentage Home Ownership By Ethnicity [2001-2016]')
#plotting data
for i in range(0,5):
    plt.scatter(x,(eth.iloc[i,5:9])*100,label = eth.iloc[i,0])
    lin_reg.fit(x_rs,(eth.iloc[i,5:9])*100)
    predicted = lin_reg.predict(x_rs)
    plt.plot(x_rs,predicted)
plt.legend(loc = "center right", bbox_to_anchor = (1.5,0.5))
plt.show()

#Regional Housing Regression Lines
reg = Type.iloc[3:12,0:11]
x = np.array([1996,2001,2006,2011,2016])
y = np.array([50,55,60,65,70,75])
#creating reg. model
lin_reg = LinearRegression(fit_intercept = True)
x_rs = x.reshape(-1,1)
#setting ticks and labels
plt.xticks(x) 
plt.yticks(y) 
plt.xlabel('Year')
plt.ylabel('Home Ownership(%)')
plt.title('Percentage Home Ownership By Region [1996-2016]')
#plotting data
for i in range(0,9):
    plt.scatter(x,(reg.iloc[i,6:11])*100,label = reg.iloc[i,0])
    lin_reg.fit(x_rs,(reg.iloc[i,6:11])*100)
    predicted = lin_reg.predict(x_rs)
    plt.plot(x_rs,predicted)
plt.legend(loc = "center right", bbox_to_anchor = (1.4,0.5))
plt.show()

# Age Regression Lines
#Accessing Required Sheet and editing
age = pd.read_excel('Housing data 2025.xlsx',sheet_name = 'Age group')
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
#ticks and labels
plt.xticks(x) 
plt.yticks(y) 
plt.xlabel('Year')
plt.ylabel('Home Ownership(%)')
plt.title('Percentage Home Ownership By Age [1996-2016]')
#plotting graphs
for i in range(0,7):
    plt.scatter(x,(age.iloc[i,6:11])*100,label = age.iloc[i,0])
    lin_reg.fit(x_rs,(age.iloc[i,6:11])*100)
    predicted = lin_reg.predict(x_rs)
    plt.plot(x_rs,predicted)
plt.legend(loc = "center right", bbox_to_anchor = (1.4,0.5))
plt.show()
#creating graph settings
plt.xticks(x) 
plt.yticks(y) 
plt.xlabel('Year')
plt.ylabel('Home Ownership(%)')
plt.title('Percentage Home Ownership By Age [1996-2016]')
#plotting data
for i in range(0,7):
    plt.scatter(x,(age.iloc[i,6:11])*100,label = age.iloc[i,0])
    p2R.fit(p2I,(age.iloc[i,6:11])*100)
    predict = p2R.predict(p2I)
    plt.plot(x,predict)
plt.legend(loc = "center right", bbox_to_anchor = (1.4,0.5))
plt.show()

# Afforability Regression Lines
#Accessing Necessary Sheets
hp = pd.read_excel('Housing data 2025.xlsx',sheet_name = 'house prices')
hp = hp.drop(hp.index[range(5)])
earn = pd.read_excel('Housing data 2025.xlsx',sheet_name = 'retail prices and earnings')

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