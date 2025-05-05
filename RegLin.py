#Importing Libraries
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

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












#Importing Libraries
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#Excel file reading and manipulating into dataframe
xl = pd.read_excel('hd25.xlsx')

#Accessing sheet related to this
Type = pd.read_excel('hd25.xlsx',sheet_name = 'Type by Region')
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