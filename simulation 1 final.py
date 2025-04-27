

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Simulation 1 Part 1
#file_path = r"C:\Users\spudu\Downloads\Housing data 2025.xlsx"
data = pd.read_excel('hd25.xlsx', sheet_name='house prices', header = 2 )
data = data[2:]

#print(data)

r = (14.75/100)/12
P_0 = 9000
    
n = np.linspace(3,360,120)
N = 360
    
C = (r/(1-((1+r)**(-N))))*P_0
P_n = (((1+r)**n)*P_0 ) - ((((1+r)**n)-1)/r)*C

House_prices = data.iloc[0:120,1]
Relative_house = (House_prices / House_prices[2]) * 10000

Equity = Relative_house - P_n
print (Equity)

plt.figure()
time = np.linspace(1974, 2003.75, 120)
over = np.linspace(2003.75,2017.25,55)
plt.plot(time, Equity,'b')
plt.plot(over,data.iloc[119:174,1],'b')
plt.xlabel("Years")
plt.ylabel("Equity")
plt.title("Total Equity in house")

total = 360 * C
print(total)

#Part 2a
House_prices2 = data[data.columns[len(data.columns)-1]]
Relative_house2 = (House_prices2 / House_prices2[2]) * 10000
Salary = 0.3* Relative_house2
#print(Salary)

#Part 2b
annual_income = pd.read_excel('hd25.xlsx', sheet_name = 'retail prices and earnings', header = 1)
avg_income = annual_income[annual_income.columns[len(annual_income.columns)-1]]
relative_inc = (avg_income / avg_income[0]) * 3000
#print (relative_inc)

plt.figure()
Years3 = np.linspace(1974,2016,174)
plt.plot(Years3, Salary, label = "Salary needed")
plt.xlabel("Years")
plt.ylabel("Salary")
plt.title("Comparison between Salary needed to afford a 90% loan and Average Annual Nominal Earnings")
Years4 = np.linspace(1974,2016, 43)
plt.plot(Years4, relative_inc, label = "£3000 Salary Adjusted with time")
plt.legend()
plt.show() 