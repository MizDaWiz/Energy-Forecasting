# Cleaning
This file checks for duplicates and null values and performs encoding for categorical features. There were 2 entries where we did not have the target values so I discarded them. I also checked for 0's in the features and found that production was 0 in 4543 rows(10% of the dataset). This led to the obvious observation that solar energy produced between sunset and sunrise is 0 or close to 0 [Hours 18 and 4]

# EDA
This file helps visualize the data and look for any improvements we can make before we train on the dataset. 

## Variation of Y with all features since they are limited
I first looked at the basic statistics of every feature in the dataset and the boxplot of the target values in buckets of the sources which gave the inference that there is a large difference in the production when we look at both sources 

![image](https://github.com/user-attachments/assets/0cf71068-3381-4986-9004-ad822f948acd)
 

Then I plotted the histogram of production values for both sources:

### Solar:<br/>

For Production < 5 (Because that is 50% of all values as we can see)
![image](https://github.com/user-attachments/assets/105563eb-1e74-40fa-bf9a-18780872ae10)

Production between 5 and 200

![image](https://github.com/user-attachments/assets/78da4647-f2f3-4c68-bb49-acd6ac1bb79b)

Production greater than 200

![image](https://github.com/user-attachments/assets/cdd401d9-0c17-47a3-9581-fac613755323)

### Wind:

Production less than 7500

![image](https://github.com/user-attachments/assets/b35dbe8a-f9f6-4f38-9528-6a109dd33b3a)

Production greater than 7500

![image](https://github.com/user-attachments/assets/b43c425d-b482-436e-9a31-63733a25982f)

This led me to the conclusion that I would benefit from using two different models when classifying the wind dataset and when classifying the solar dataset due to the large order of magnitude difference and the seasonanility in solar data which is absent in wind data that I will see further ahead <br/>

![image](https://github.com/user-attachments/assets/4535b05e-dab6-45a5-909b-954068df22fe)


From the number of observations we had in each year, to plot the Monthly, Daily and Hourly production, I only considered 2021 and 2022 for homogenity 

### Monthly Production:

![image](https://github.com/user-attachments/assets/72b80adb-e9e4-444e-9cc5-545a60dba441)


### Daily Production:

![image](https://github.com/user-attachments/assets/3cc0417f-e751-4f79-b95b-bb52429cc67b)

### Hourly Production: 

![image](https://github.com/user-attachments/assets/5fc4f827-f00c-4c66-8f1d-be46845886dd)

Solar Production values closely resembles a gaussian curve with respect to
months and with respect to hours between 6 to 21 (almost like a sine
function)  <br/>

Wind Production also shows a gaussian-like curve shape if we consider the cycle to
be from month 6 to month 5 or hours from 10hrs to 9 hrs <br/>

There is a difference in behavior of the variation of production with features when the sources changes <br/>

With these observations, I decided to add 2 new features as sin functions for hour and month which cover a pi interval across the range of those values and concluded that I should use two different models to predict energy for either source<br/>

I then observed the correlation matrix of production with features for both sources separately and eliminated features with <5% correlation and exported the data to 2 new datasets for wind and solar <br/>

### For Solar: 
<img width="148" alt="image" src="https://github.com/user-attachments/assets/42044a85-1f57-485d-97fc-3358d14eca78">

### For Wind: 
<img width="146" alt="image" src="https://github.com/user-attachments/assets/0cf76c53-061a-468e-8cc6-f76e8c071681">


# Baseline Models
We train some standard machine learning models to get a baseline for model performance

# LSTM 
We implement an LSTM for our prediction tasks to get the best model performance. I found that sequence length 6 was best which was in line with my expectation due to the nature of the hourly production graph 
