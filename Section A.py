# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 14:58:06 2025

@author: FiercePC
"""

#Gannt Chart
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#Listing Data
tasks = ['Project Planning','Short Presentation','DA Renting','DA Ownership','Comparing','Regression Lines','Sim 1','Sim 2','Report','Bibliography','Main Presentation','Proofread']
start_dates = ['2025-04-01','2025-04-06','2025-04-05','2025-04-05','2025-04-12','2025-04-05','2025-04-15','2025-04-15','2025-04-05','2025-04-30','2025-04-23','2025-05-02']
end_dates = ['2025-04-06','2025-04-08','2025-04-12','2025-04-12','2025-04-15','2025-04-13','2025-04-22','2025-04-22','2025-05-01','2025-05-01','2025-04-30','2025-05-08']
person = ['Everyone','Everyone','Abi + Keeya','Gareema + SamW','Abi,Keeya,Gareema + SamW','Paul,Suzy + SamE','Paul + SamW','Abi,Keeya + Gareema','SamE + Suzy','SamE + Suzy','Everyone','Everyone']
# dictionary of persons and colours
person_colours = {'Everyone':'tab:grey','Abi + Keeya':'tab:blue','Gareema + SamW':'tab:red','Abi,Keeya,Gareema + SamW':'tab:purple','Paul,Suzy + SamE':'tab:olive','Paul + SamW':'tab:pink','Abi,Keeya + Gareema':'tab:cyan','SamE + Suzy':'tab:green'}
               
#Creating Dataframe
data = {'Tasks':tasks,'Start Dates':start_dates,'End Dates':end_dates,'Person':person}
df = pd.DataFrame(data)
# convert the start and end dates to datetimes
df['Start Dates'] = pd.to_datetime(df['Start Dates'])
df['End Dates'] = pd.to_datetime(df['End Dates'])
#Sorting Dates
df = df.sort_values(by = 'Start Dates', ascending = True, ignore_index = True)

# initialise a list of patches (of colour)
patches = []
# append colour patches from person_colours to the list of patches
for person in person_colours:
    patches.append(mpatches.Patch(color = person_colours[person]))

# Calculate total project duration
duration = (df['End Dates'].max()) - (df['Start Dates'].min())
# add 1 day to include the last day as part of the task
df['Task Duration'] = df['End Dates'] - df['Start Dates'] + pd.Timedelta(days = 1)
                                
#create a figure
fig, ax = plt.subplots()
ax.xaxis_date()
for index, row in df.iterrows():
    plt.barh(y = row['Tasks'], width = row['Task Duration'], left = row['Start Dates'], color = person_colours[row['Person']],alpha = 1)
    
ax.set_title('Gantt Chart Project Plan')
ax.set_xlabel('Date')
ax.set_ylabel('Task')
ax.set_xlim(df['Start Dates'].min(),df['End Dates'].max())
fig.gca().invert_yaxis()

# create a date range for the ticks in 5 day intervals
date_range = pd.date_range(start = pd.to_datetime("2025-04-01"),end = pd.to_datetime("2025-05-11"),freq = "4D")
ax.set_xticks(date_range)
# rotate the ticks
ax.tick_params(axis = 'x', labelrotation = 45)
# plots gridlines for the x axis
ax.xaxis.grid(True, alpha=0.5)

#legend
ax.legend(handles=patches, labels=person_colours.keys(), fontsize=10,bbox_to_anchor = (1,0.5))

print(df)