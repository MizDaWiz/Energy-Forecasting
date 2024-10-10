# Source Code

## Cleaning
This file checks for duplicates and null values and performs encoding for categorical features. There were 2 entries where we did not have the target values so I discarded them. I also checked for 0's in the features and found that production was 0 in 4543 rows(10% of the dataset). This led to the obvious observation that solar energy produced between sunset and sunrise is 0 or close to 0 [Hours 18 and 4]

## EDA
This file helps visualize the data and look for any improvements we can make before we train on the dataset. 

I first looked at the basic statistics of every feature in the dataset and the boxplot of the target values in buckets of the sources which gave the inference that there is a large difference in the production when we look at both sources 

**img 

Then I plotted the histogram of production values for both sources:

Solar:

Wind:

This led me to the conclusion that I would benefit from using two different models when classifying the wind dataset and when classifying the solar dataset due to the large order of magnitude difference and the seasonanility in solar data which is absent in wind data that I will see further ahead 

From the number of observations we had in each year, to plot the Monthly, Daily and Hourly production, I only considered 2021 and 2022 for homogenity 

Yearly Production:

**img 
Monthly Production:

**img
Daily Production:

**img

Solar Production closely resembles a gaussian curve with respect to
months and with respect to hours between 6 to 21 (almost like a sine
function)  

Wind Production also shows a gaussian curve if we consider the cycle to
be from month 6 to month 5 or hours from 10hrs to 9 hrs

There is a difference in behavior of the variation of production with features when the sources changes 

So I added 2 new features as sin functions for hour and month which cover a pi interval across the range of those values and concluded that I should use two different models to predict energy for either source

I then observed the correlation matrix of both features and eliminated those with <5% correlation or repeats and exported the data to 2 new datasets for wind and solar 




## Baseline Models
We train some standard machine learning models to get a baseline for model performance

## LSTM 
We implement an LSTM for our prediction tasks to get the best model performance
