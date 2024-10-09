# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 17:15:32 2024

@author: mizaa
"""
#Imports 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
path = 'C:/Users/mizaa/Desktop/Energy Forecasting/data/cleaneddata.csv'
# %%Load the data
df=pd.read_csv(path)
df.info()
k=df.describe()

# %% BoxPlot Production By Source

plt.figure(figsize=(8, 6))
df.boxplot(column='Production', by='Source', grid=False)
# =============================================================================
# tem=df[df['Production']>5000]
# tem.boxplot(column='Production', grid=False)
# # Adding titles and labels
# =============================================================================
plt.title('Box Plot of Production')
plt.suptitle('')  # Suppress the default title to improve aesthetics
plt.xlabel('Source')
plt.ylabel('Production')

# Show plot
plt.show()


# %% Monthly Production By Source (bar graph)

# Filter for specific years (e.g., 2021 and 2022)
years_to_plot = [2021, 2022]
filtered_df = df[df['Year'].isin(years_to_plot)]

# Group by month and Source, summing the Production
total_production = filtered_df.groupby(['month_number', 'Source'])['Production'].sum().unstack()

# Create a bar plot
total_production.plot(kind='bar', figsize=(12, 6))

# Adding titles and labels
plt.title('Monthly Production by Source for Selected Years')
plt.xlabel('Month')
plt.ylabel('Total Production')
plt.xticks(rotation=45)
plt.legend(title='Source')
plt.grid(axis='y')

# Show plot
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()


# %% Total Production daywise 
years_to_plot = [2021, 2022]
filtered_df = df[df['Year'].isin(years_to_plot)]

# Group by daynumber and sum the Production
total_production = filtered_df.groupby('dayOfYear')['Production'].sum()

# Create a line plot
plt.figure(figsize=(12, 6))
plt.plot(total_production.index, total_production.values, linestyle='-', color='b')

# Adding titles and labels
plt.title('Total Production by Day Number for Selected Years')
plt.xlabel('Day Number')
plt.ylabel('Total Production')
plt.xticks(range(0, 390, 30))  # Show ticks every 25 days for clarity
plt.grid()

# Show plot
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()

# %% Total Production For each source daywise
years_to_plot = [2021, 2022]
filtered_df = df[df['Year'].isin(years_to_plot)]

#Separating data sourcewise
filtered_df1 = filtered_df[filtered_df['Source']==1]
filtered_df0 = filtered_df[filtered_df['Source']==0]

# Group by daynumber and sum the Production
total_production1 = filtered_df1.groupby('dayOfYear')['Production'].sum()
total_production0 = filtered_df0.groupby('dayOfYear')['Production'].sum()

# Create a line plot
plt.figure(figsize=(12, 6))
plt.plot(total_production0.index, total_production0.values, linestyle='-', color='b')
plt.plot(total_production1.index, total_production1.values, linestyle='-', color='r')

# Adding titles and labels
plt.title('Total Production by Day Number for Selected Years')
plt.xlabel('Day Number')
plt.ylabel('Total Production')
plt.xticks(range(0, 390, 30))  # Show ticks every 25 days for clarity
plt.grid()

# Show plot
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()

# %% Hourly Production by source

# Group by Hour and Source, summing the Production
total_production = df.groupby(['Hour', 'Source'])['Production'].sum().unstack()

# Create a bar plot
total_production.plot(kind='bar', figsize=(12, 6))

# Adding titles and labels
plt.title('Total Production by Hour for Each Source')
plt.xlabel('Hour of the Day')
plt.ylabel('Total Production')
plt.xticks(rotation=0)  # Rotate x-axis labels for better visibility
plt.legend(title='Source')
plt.grid(axis='y')

# Show plot
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()

# %% Plot Histogram of Production
filtered = df#[(df['Production'] >6000) ]
plt.figure(figsize=(10, 6))
plt.hist(filtered['Production'], bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram of Production')
plt.xlabel('Production')
plt.ylabel('Frequency')
#plt.yticks(range(0,2000,100))
plt.grid(True)
plt.show()
# %% Sourcewise data
y = df[df['Source']==1]
y2 = df[df['Source']==0]
# %% Sourcewise Histogram

# Wind
#yz1= y[y['Production']>7500]
yz1= y[y['Production']<7500]
plt.figure(figsize=(10, 6))
plt.hist(yz1['Production'], bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram of Production')
plt.xlabel('Production')
plt.ylabel('Frequency')
#plt.yticks(range(0,2000,100))
plt.grid(True)
plt.show()
# %%Solar

yz2 = y2[y2['Production']<10]
#yz2= yz2[yz2['Production']<200]
plt.figure(figsize=(10, 6))
plt.hist(yz2['Production'], bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram of Production')
plt.xlabel('Production')
plt.ylabel('Frequency')
#plt.yticks(range(0,2000,100))
plt.grid(True)
plt.show()

# %% Check Sequence completeness


# %% Drop Extra Columns
df=df.drop(columns=['Date and Hour','Date'])

# %% Check corr with experimental features

df['month_sinhalf'] = np.sin(np.pi * df['Hour'] / 12)
df['hour_sinhalf'] = np.sin(np.pi * df['Hour'] / 23)

temp0=df[df['Source']==0]
temp1=df[df['Source']==1]
corr0=temp0.corr()
corr1=temp1.corr()
p=df.corr()
# %%

k00 = df.describe()
k0 = temp0.describe()
k1 = temp1.describe()
# %% Dropping low corellation features

finaldata = df.drop(columns=['dayName', 'Year'])
finaldata0 = temp0.drop(columns=['dayName', 'Year', 'Hour','month_number', 'Source']) #Try dropping dayofYear
finaldata1 = temp1.drop(columns=['dayName', 'Year',  'month_sinhalf', 'Source' ])
# %% Store Data

finaldata.to_csv('C:/Users/mizaa/Desktop/Energy Forecasting/Data/finaldata.csv',index=False )
finaldata0.to_csv('C:/Users/mizaa/Desktop/Energy Forecasting/Data/finaldata_0.csv',index=False )
finaldata1.to_csv('C:/Users/mizaa/Desktop/Energy Forecasting/Data/finaldata_1.csv',index=False )