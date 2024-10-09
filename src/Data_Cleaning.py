# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 22:55:29 2024
@author: mizaa
# %%
"""
# %%
# Imports

import pandas as pd 
import calendar
from sklearn.preprocessing import LabelEncoder
path = 'C:/Users/mizaa/Desktop/Energy Forecasting/data/intermittent-renewables-production-france.csv'
# %% Load the data
df = pd.read_csv(path)
df.info()

# %%Delete any duplicates, there were no duplicates
df.drop_duplicates()
df.info()

# %%Encode Month Names into numbers

# Create a dictionary to map month names to month numbers
month_map = {'January':1,'February':2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7,'August':8, 'September':9, 'October':10, 'November':11, 'December':12}

# Convert the month names to month numbers
df['month_number'] = df['monthName'].map(month_map)

# %%Label encoding
label_encoder = LabelEncoder()
df['Source'] = label_encoder.fit_transform(df['Source'])
df['dayName'] = label_encoder.fit_transform(df['dayName'])



# %% Check which columns have null values
df.isnull().any()
# %% Check and Drop rows with null values 
nanrows = df[df['Production'].isnull()]
df_nonull = df[df['Production'].notna()]
df_nonull.info()


# %% Check for 0 values
df.eq(0).any()
zerorows = df[df['Production'].eq(0)]

# %% Analysis of Zeroes
zerostarts = zerorows['StartHour'].value_counts().sort_values()
zeroends = zerorows['EndHour'].value_counts().sort_values()

# %%Extract Hour Number from StartHour & Year from Date & hour
df_nonull['Hour']=df_nonull.StartHour.str[:2]
df_nonull['Hour']=pd.to_numeric(df_nonull['Hour'])
df_nonull['Year']=df_nonull.Date.str[6:]
df_nonull['Year']=pd.to_numeric(df_nonull['Year'])

# %% Dropping Redundant Features
final_df = df_nonull.drop(columns=['StartHour','EndHour','monthName'])

final_df.info()


# %%Get Clean Data
final_df.to_csv('C:/Users/mizaa/Desktop/Energy Forecasting/data/cleaneddata.csv', index=False)






