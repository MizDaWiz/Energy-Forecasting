# Energy-Forecasting

## Data
The dataset contained energy production values for every hour of every day between 2020 and 2023 along with the source of production(solar or wind). The task was to train a model to be able to forecast production of energy 

## Features 
1. Energy Source:
2. StartTime of Production: xx:xx:xx format
3. EndTime of Production: xx:xx:xx format
4. Date of Production
5. DayName of Production
6. DayNum(of Year) of Production
7. Month of Production

## Process
After cleaning the data, pre-processing and engineering features, I  trained a few basic models on the dataset to get a baseline for the RMSE. Then I trained an LSTM network on the data in tensorflow.  
I've gone into depth of the decision making in the pre-processing and feature selection inside the src files 

