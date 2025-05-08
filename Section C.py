#Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

###SIMULATION 1
#Simulation 1 Part 1
#reading data from sheet
data = pd.read_excel('Housing data 2025.xlsx', sheet_name='house prices', header = 2 )
data = data[2:]

#setting values used in equations
r = (14.75/100)/12
P_0 = 9000
n = np.linspace(3,360,120)
N = 360
#creating formulae
C = (r/(1-((1+r)**(-N))))*P_0
P_n = (((1+r)**n)*P_0 ) - ((((1+r)**n)-1)/r)*C

#working out house cost
House_prices = data.iloc[0:120,1]
Relative_house = (House_prices / House_prices[2]) * 10000
#Calculating equity
Equity = Relative_house - P_n
print (Equity)

#plotting data
plt.figure()
time = np.linspace(1974, 2003.75, 120)
over = np.linspace(2003.75,2017.25,55)
plt.plot(time, Equity,'b')
plt.plot(over,data.iloc[119:174,1],'b')
plt.xlabel("Years")
plt.ylabel("Equity")
plt.title("Total Equity in house")
#calculating total paid
total = 360 * C
print(total)

#Part 2a
House_prices2 = data[data.columns[len(data.columns)-1]]
Relative_house2 = (House_prices2 / House_prices2[2]) * 10000
Salary = 0.3* Relative_house2

#Part 2b
annual_income = pd.read_excel('Housing data 2025.xlsx', sheet_name = 'retail prices and earnings', header = 1)
avg_income = annual_income[annual_income.columns[len(annual_income.columns)-1]]
relative_inc = (avg_income / avg_income[0]) * 3000
#plotting data
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

###SIMULATION 2
''' SIMULATION 2 '''

''' Q1 '''
# assumption four - stamp duty 
def calculate_stamp_duty(price):
    if price < 125000:
        return 0
    elif price < 250000:
        return (price - 125000) * 0.02
    elif price < 925000:
        return (250000 - 125000) * 0.02 + (price - 250000) * 0.05
    elif price < 1500000:
        return (250000 - 125000) * 0.02 + (925000 - 250000) * 0.05 + (price - 925000) * 0.10
    else:
        return (250000 - 125000) * 0.02 + (925000 - 250000) * 0.05 + (1500000 - 925000) * 0.10 + (price - 1500000) * 0.12

# assumption seven - adjust for inflation
def adjust_for_inflation(value_2016, rpi_current, rpi_2016):
    return value_2016 * (rpi_current / rpi_2016)

def remaining_mortgage(P0, r, n, C):
    mortgage_balance = (((1 + r)**n) * P0) - ((((1 + r)**n) - 1) / r) * C
    return mortgage_balance
    
def simulation():
    # for the graph
    equity_totals = []
    years = []

    # initial values 
    x = 0.2  # 20% of accumulated equity
    
    buying_year = 1974
    house_price = 10000
    deposit = 1000
    salary = 3000
    mortgage = house_price - deposit
    mortgage_term = 30
    
    invested_equity = 0
    
    # extract the relevant data
    # assumption one - property values increase in line with average house prices
    df_1 = pd.read_excel('Housing data 2025.xlsx', sheet_name='house prices', header = 2)
    df_1 = df_1[2:]
    house_prices = df_1.iloc[0:177,1]
    relative_house = (house_prices / house_prices[2]) * 10000
    yearly_price = relative_house[::4].reset_index(drop=True)
    
    # assumption two - salary increases like in implementation 1
    df_2 = pd.read_excel('Housing data 2025.xlsx', sheet_name = 'retail prices and earnings', header = 1)
    avg_income = df_2[df_2.columns[len(df_2.columns)-1]]
    relative_inc = (avg_income / avg_income[0]) * 3000
    relative_inc = relative_inc.reset_index(drop=True)
    
    # get relevant data for assumption three
    df_3 = pd.read_excel('Housing data 2025.xlsx', sheet_name = 'interest rates', header = 0)
    df_3 = df_3.drop(df_3.columns[[0]], axis=1)
    bank_rates = df_3.iloc[0:512, 1:2] 
    
    # get bank interests and mortgage interests for every january
    bank_rates = bank_rates[::12].reset_index(drop=True)
    mortgage_rates = df_3.iloc[0:512, 2:3]
    mortgage_rates = mortgage_rates[::12].reset_index(drop=True)
    
    # initial mortgage interest
    mortgage_interest = mortgage_rates.iloc[0].values[0] / 100
    
    # rpi retrieval
    rpi_2016 = df_2.loc[df_2['Year'] == 2016, 'Retail Price Index (2010 = 100)'].values[0]
    
    mortgage_paid = 0
    
    # simulation starts
    for i in range(0, 41):
        current_year = 1974 + i
        rpi_current = df_2.loc[df_2['Year'] == current_year, 'Retail Price Index (2010 = 100)'].values[0]
        
        house_price = yearly_price[i]
        salary = relative_inc[i]
        bank_interest = bank_rates.iloc[i].values[0] / 100
        
        # calculations for mortgage payments using implementation 1 formula
        mortgage_payment = (mortgage_interest / (1 - (1 + mortgage_interest) ** (-mortgage_term))) * mortgage
        n = current_year - buying_year
        mortgage = remaining_mortgage(mortgage, mortgage_interest, n, mortgage_payment)
        
        mortgage_paid += mortgage_payment
        
        # record total equity this year
        invested_equity *= (1 + bank_interest)
        #print(invested_equity)
        equity = house_price + invested_equity - mortgage
        
        # for the graph
        equity_totals.append(equity)
        years.append(current_year)

        # assumption eight - bought and sold in Q1
        if (i % 5 == 0) and (i != 0):
            # update year for mortgage payments
            buying_year = 1974 + i
            
            # assumption five - selling fees
            sale_price = house_price
            estate_fees = 0.02 * sale_price
            legal_fees = adjust_for_inflation(400, rpi_current, rpi_2016)
            selling_fees = legal_fees + estate_fees + (0.004 * sale_price)
            accumulated_equity = sale_price - mortgage - selling_fees
            
            if current_year == 2014:
                ''' Q3 - plot equity in 2014 as a function of x '''
                final_equities = []
                plot_xs = np.linspace(0, 100, 101)
                xs = np.linspace(0.1, 1, 101)
                temp_equity = accumulated_equity
                
                for x in xs:
                    temp_deposit = temp_equity * x
                    final_equity = accumulated_equity - temp_deposit
                    final_equities.append(final_equity) 
                    
                # plot equity over time
                plt.figure(figsize=(10,5))
                plt.plot(plot_xs, final_equities, label="Final Equity (£)")
                plt.xlabel("x% of Equity Used As Deposit")
                plt.ylabel("Final Equity (£)")
                plt.title("Final Equity Based On How Much Will Be Used As Deposit")
                plt.show()  
                
            # new purchase for a house that costs the same as the house we just sold
            
            # calculate how much of accumulated equity to use for buying the new house
            possible_xs = np.linspace(0, 0.7, 71)
            for x in possible_xs:
                if ((salary * 3) + (accumulated_equity * x)) >= house_price:
                    if x == 0 or x < 0.1:
                        # fix x if we do have enough money to buy the house
                        x = 0.1
                    break
            deposit = accumulated_equity * x
            loan = 3 * salary
            budget = deposit + loan
            # the new house has the same price as the last one
            house_price = yearly_price[i]
            
            # assumption six - cost of buying a house depending on the current year
            buying_fees = adjust_for_inflation(1500, rpi_current, rpi_2016)
            buying_fees += 0.003 * house_price
            stamp_duty = calculate_stamp_duty(house_price)
            total_price = house_price + buying_fees + stamp_duty
            
            # invest the remaining money
            invested_equity += budget - total_price
            
            # update new mortgage and its rate
            mortgage = house_price - deposit
            mortgage_interest = mortgage_rates.iloc[i].values[0] / 100
    
    ''' Q2 '''
    # plot equity over time
    plt.figure(figsize=(10,5))
    plt.plot(years, equity_totals, label="Total Equity (£)", marker='o')
    plt.xlabel("Year")
    plt.ylabel("Equity (£)")
    plt.title("Accumulated Equity When Buying Every 5 Years [1974-2014]")
    plt.show()  
    
    print(f'The amount of mortgage paid is £{round(mortgage_paid, 2)}')
    
simulation()

''' Q4 (10 Year Intervals) '''
# assumption four - stamp duty 
def calculate_stamp_duty(price):
    if price < 125000:
        return 0
    elif price < 250000:
        return (price - 125000) * 0.02
    elif price < 925000:
        return (250000 - 125000) * 0.02 + (price - 250000) * 0.05
    elif price < 1500000:
        return (250000 - 125000) * 0.02 + (925000 - 250000) * 0.05 + (price - 925000) * 0.10
    else:
        return (250000 - 125000) * 0.02 + (925000 - 250000) * 0.05 + (1500000 - 925000) * 0.10 + (price - 1500000) * 0.12

# assumption seven - adjust for inflation
def adjust_for_inflation(value_2016, rpi_current, rpi_2016):
    return value_2016 * (rpi_current / rpi_2016)

def remaining_mortgage(P0, r, n, C):
    mortgage_balance = (((1 + r)**n) * P0) - ((((1 + r)**n) - 1) / r) * C
    return mortgage_balance
    
def simulation():
    # for the graph
    equity_totals = []
    years = []

    # initial values 
    x = 0.2  # 20% of accumulated equity
    
    buying_year = 1974
    house_price = 10000
    deposit = 1000
    salary = 3000
    mortgage = house_price - deposit
    mortgage_term = 30
    
    invested_equity = 0
    
    # extract the relevant data
    # assumption one - property values increase in line with average house prices
    df_1 = pd.read_excel('Housing data 2025.xlsx', sheet_name='house prices', header = 2)
    df_1 = df_1[2:]
    house_prices = df_1.iloc[0:177,1]
    relative_house = (house_prices / house_prices[2]) * 10000
    yearly_price = relative_house[::4].reset_index(drop=True)
    
    # assumption two - salary increases like in implementation 1
    df_2 = pd.read_excel('Housing data 2025.xlsx', sheet_name = 'retail prices and earnings', header = 1)
    avg_income = df_2[df_2.columns[len(df_2.columns)-1]]
    relative_inc = (avg_income / avg_income[0]) * 3000
    relative_inc = relative_inc.reset_index(drop=True)
    
    # get relevant data for assumption three
    df_3 = pd.read_excel('Housing data 2025.xlsx', sheet_name = 'interest rates', header = 0)
    df_3 = df_3.drop(df_3.columns[[0]], axis=1)
    bank_rates = df_3.iloc[0:512, 1:2] 
    
    # get bank interests and mortgage interests for every january
    bank_rates = bank_rates[::12].reset_index(drop=True)
    mortgage_rates = df_3.iloc[0:512, 2:3]
    mortgage_rates = mortgage_rates[::12].reset_index(drop=True)
    
    # initial mortgage interest
    mortgage_interest = mortgage_rates.iloc[0].values[0] / 100
    
    # rpi retrieval
    rpi_2016 = df_2.loc[df_2['Year'] == 2016, 'Retail Price Index (2010 = 100)'].values[0]
    
    ''' Q2 '''
    mortgage_paid = 0
    
    # simulation starts
    for i in range(0, 41):
        current_year = 1974 + i
        rpi_current = df_2.loc[df_2['Year'] == current_year, 'Retail Price Index (2010 = 100)'].values[0]
        
        house_price = yearly_price[i]
        salary = relative_inc[i]
        bank_interest = bank_rates.iloc[i].values[0] / 100
        
        # calculations for mortgage payments using implementation 1 formula
        mortgage_payment = (mortgage_interest / (1 - (1 + mortgage_interest) ** (-mortgage_term))) * mortgage
        n = current_year - buying_year
        mortgage = remaining_mortgage(mortgage, mortgage_interest, n, mortgage_payment)
        
        mortgage_paid += mortgage_payment
        
        # record total equity this year
        invested_equity *= (1 + bank_interest)
        #print(invested_equity)
        equity = house_price + invested_equity - mortgage
        
        # for the graph
        equity_totals.append(equity)
        years.append(current_year)

        # assumption eight - bought and sold in Q1
        if (i % 10 == 0) and (i != 0):
            # update year for mortgage payments
            buying_year = 1974 + i
            
            # assumption five - selling fees
            sale_price = house_price
            estate_fees = 0.02 * sale_price
            legal_fees = adjust_for_inflation(400, rpi_current, rpi_2016)
            selling_fees = legal_fees + estate_fees + (0.004 * sale_price)
            accumulated_equity = sale_price - mortgage - selling_fees
            
            # new purchase for a house that costs the same as the house we just sold
            
            # calculate how much of accumulated equity to use for buying the new house
            possible_xs = np.linspace(0, 0.7, 71)
            for x in possible_xs:
                if ((salary * 3) + (accumulated_equity * x)) >= house_price:
                    if x == 0 or x < 0.1:
                        # fix x if we do have enough money to buy the house
                        x = 0.1
                    break
            deposit = accumulated_equity * x
            loan = 3 * salary
            budget = deposit + loan
            # the new house has the same price as the last one
            house_price = yearly_price[i]
            
            # assumption six - cost of buying a house depending on the current year
            buying_fees = adjust_for_inflation(1500, rpi_current, rpi_2016)
            buying_fees += 0.003 * house_price
            stamp_duty = calculate_stamp_duty(house_price)
            total_price = house_price + buying_fees + stamp_duty
            
            # invest the remaining money
            invested_equity += budget - total_price
            
            # update new mortgage and its rate
            mortgage = house_price - deposit
            mortgage_interest = mortgage_rates.iloc[i].values[0] / 100
    
    ''' Q2 '''
    # plot equity over time
    plt.figure(figsize=(10,5))
    plt.plot(years, equity_totals, label="Total Equity (£)", marker='o')
    plt.xlabel("Year")
    plt.ylabel("Equity (£)")
    plt.title("Accumulated Equity When Buying Every 10 Years [1974-2014]")
    plt.show()  
    
    print(f'The amount of mortgage paid is £{round(mortgage_paid, 2)}')
    
simulation()

''' Q4 (3 Year Intervals) '''
# assumption four - stamp duty 
def calculate_stamp_duty(price):
    if price < 125000:
        return 0
    elif price < 250000:
        return (price - 125000) * 0.02
    elif price < 925000:
        return (250000 - 125000) * 0.02 + (price - 250000) * 0.05
    elif price < 1500000:
        return (250000 - 125000) * 0.02 + (925000 - 250000) * 0.05 + (price - 925000) * 0.10
    else:
        return (250000 - 125000) * 0.02 + (925000 - 250000) * 0.05 + (1500000 - 925000) * 0.10 + (price - 1500000) * 0.12

# assumption seven - adjust for inflation
def adjust_for_inflation(value_2016, rpi_current, rpi_2016):
    return value_2016 * (rpi_current / rpi_2016)

def remaining_mortgage(P0, r, n, C):
    mortgage_balance = (((1 + r)**n) * P0) - ((((1 + r)**n) - 1) / r) * C
    return mortgage_balance
    
def simulation():
    # for the graph
    equity_totals = []
    years = []

    # initial values 
    x = 0.2  # 20% of accumulated equity
    
    buying_year = 1974
    house_price = 10000
    deposit = 1000
    salary = 3000
    mortgage = house_price - deposit
    mortgage_term = 30
    
    invested_equity = 0
    
    # extract the relevant data
    # assumption one - property values increase in line with average house prices
    df_1 = pd.read_excel('Housing data 2025.xlsx', sheet_name='house prices', header = 2)
    df_1 = df_1[2:]
    house_prices = df_1.iloc[0:177,1]
    relative_house = (house_prices / house_prices[2]) * 10000
    yearly_price = relative_house[::4].reset_index(drop=True)
    
    # assumption two - salary increases like in implementation 1
    df_2 = pd.read_excel('Housing data 2025.xlsx', sheet_name = 'retail prices and earnings', header = 1)
    avg_income = df_2[df_2.columns[len(df_2.columns)-1]]
    relative_inc = (avg_income / avg_income[0]) * 3000
    relative_inc = relative_inc.reset_index(drop=True)
    
    # get relevant data for assumption three
    df_3 = pd.read_excel('Housing data 2025.xlsx', sheet_name = 'interest rates', header = 0)
    df_3 = df_3.drop(df_3.columns[[0]], axis=1)
    bank_rates = df_3.iloc[0:512, 1:2] 
    
    # get bank interests and mortgage interests for every january
    bank_rates = bank_rates[::12].reset_index(drop=True)
    mortgage_rates = df_3.iloc[0:512, 2:3]
    mortgage_rates = mortgage_rates[::12].reset_index(drop=True)
    
    # initial mortgage interest
    mortgage_interest = mortgage_rates.iloc[0].values[0] / 100
    
    # rpi retrieval
    rpi_2016 = df_2.loc[df_2['Year'] == 2016, 'Retail Price Index (2010 = 100)'].values[0]
    
    mortgage_paid = 0
    
    # simulation starts
    for i in range(0, 41):
        current_year = 1974 + i
        rpi_current = df_2.loc[df_2['Year'] == current_year, 'Retail Price Index (2010 = 100)'].values[0]
        
        house_price = yearly_price[i]
        salary = relative_inc[i]
        bank_interest = bank_rates.iloc[i].values[0] / 100
        
        # calculations for mortgage payments using implementation 1 formula
        mortgage_payment = (mortgage_interest / (1 - (1 + mortgage_interest) ** (-mortgage_term))) * mortgage
        n = current_year - buying_year
        mortgage = remaining_mortgage(mortgage, mortgage_interest, n, mortgage_payment)
        
        mortgage_paid += mortgage_payment
        
        # record total equity this year
        invested_equity *= (1 + bank_interest)
        #print(invested_equity)
        equity = house_price + invested_equity - mortgage
        
        # for the graph
        equity_totals.append(equity)
        years.append(current_year)

        # assumption eight - bought and sold in Q1
        if (i % 3 == 0) and (i != 0):
            # update year for mortgage payments
            buying_year = 1974 + i
            
            # assumption five - selling fees
            sale_price = house_price
            estate_fees = 0.02 * sale_price
            legal_fees = adjust_for_inflation(400, rpi_current, rpi_2016)
            selling_fees = legal_fees + estate_fees + (0.004 * sale_price)
            accumulated_equity = sale_price - mortgage - selling_fees
            
            # new purchase for a house that costs the same as the house we just sold
            
            # calculate how much of accumulated equity to use for buying the new house
            possible_xs = np.linspace(0, 0.7, 71)
            for x in possible_xs:
                if ((salary * 3) + (accumulated_equity * x)) >= house_price:
                    if x == 0 or x < 0.1:
                        # fix x if we do have enough money to buy the house
                        x = 0.1
                    break
            
            deposit = accumulated_equity * x
            loan = 3 * salary
            budget = deposit + loan
            # the new house has the same price as the last one
            house_price = yearly_price[i]
            
            # assumption six - cost of buying a house depending on the current year
            buying_fees = adjust_for_inflation(1500, rpi_current, rpi_2016)
            buying_fees += 0.003 * house_price
            stamp_duty = calculate_stamp_duty(house_price)
            total_price = house_price + buying_fees + stamp_duty
            
            # invest the remaining money
            invested_equity += budget - total_price
            
            # update new mortgage and its rate
            mortgage = house_price - deposit
            mortgage_interest = mortgage_rates.iloc[i].values[0] / 100
    
    # plot equity over time
    plt.figure(figsize=(10,5))
    plt.plot(years, equity_totals, label="Total Equity (£)", marker='o')
    plt.xlabel("Year")
    plt.ylabel("Equity (£)")
    plt.title("Accumulated Equity When Buying Every 3 Years [1974-2014]")
    plt.show()  
    
    print(f'The amount of mortgage paid is £{round(mortgage_paid, 2)}')
    
simulation()