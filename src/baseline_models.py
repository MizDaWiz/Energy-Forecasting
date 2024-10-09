# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 22:21:17 2024
@author: mizaa
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, r2_score
import pickle 
csv_file_path = 'C:/Users/mizaa/Desktop/Energy Forecasting/data/finaldata.csv'
csv_file_path0 = 'C:/Users/mizaa/Desktop/Energy Forecasting/data/finaldata_0.csv'
csv_file_path1 = 'C:/Users/mizaa/Desktop/Energy Forecasting/data/finaldata_1.csv'

scaler = StandardScaler()  

# %% Load DataFrame
#data = pd.read_csv(csv_file_path)

#data = pd.read_csv(csv_file_path0)
data = pd.read_csv(csv_file_path1)


n_fdim = data.shape[1]

# %%# %% Grid Search For RF 
def tune_with_grid_search(features, targets, model):
    param_grid = {
 # single model 
        # 'n_estimators': [ 20, 30, 60, 75, 100, 150, 200, 250,300],
        # 'learning_rate': [0.01, 0.1, 0.2],
        # 'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 12,15],
        # 'max_samples': [0.5, 0.75, 0.6, 0.9, 0.8],
        # 'max_features': [2,3,4,5,6]
#Solar model 
      #  'n_estimators': [ 10, 50, 100, 150, 200],
      # # 'learning_rate': [0.01, 0.1, 0.2],
      #  'max_depth': [2, 4, 6,  9, 15],
      #  'max_samples': [0.25, 0.5, 0.75, 0.9],
      #  'max_features': [2,3,4,5]
# Wind model 
       'n_estimators': [50, 100, 150, 175, 200],
      # 'learning_rate': [0.01, 0.1, 0.2],
       'max_depth': [ 2, 3, 4, 7, 8],
       'max_samples': [.25, 0.5, 0.75, 0.6],
       'max_features': [2,3,4]
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(features, targets)
    print(f'Best Parameters from Grid Search: {grid_search.best_params_}')
    return grid_search.best_estimator_


# %%Randomized Search For XGB

def tune_with_random_search(features, targets, model):
    param_distributions = {
        'n_estimators': [int(x) for x in np.linspace(start=4, stop=60, num=45)],
        'learning_rate': np.linspace(0.001, 0.1, num=10000),
        'max_depth': [int(x) for x in np.linspace(2, 8, num=6)]
    }

    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_distributions, n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
    random_search.fit(features, targets)
    print(f'Best Parameters from Randomized Search: {random_search.best_params_}')
    return random_search.best_estimator_



# %% Non-normalized Data
target = data['Production']
data_1 = data.drop(columns=['Production'])
features1_train , features1_test, target_train, target_test = train_test_split(data_1,target, test_size=0.2, random_state=42)
# %%Linear Regression
lr_model1 = LinearRegression()
lr_model1.fit(features1_train, target_train)

pred = lr_model1.predict(features1_test)

test_pred_lr = pd.DataFrame(pred)
rmse1_lr = root_mean_squared_error(target_test, test_pred_lr)
r2_lr1 = r2_score(target_test, test_pred_lr)

# %%Random Forest

rf_model1 = RandomForestRegressor(random_state=42)
best_rf_grid1 = tune_with_grid_search(features1_train ,target_train, rf_model1)

pred = best_rf_grid1.predict(features1_test)

test_pred_rf = pd.DataFrame(pred)
rmse1_rf = root_mean_squared_error(target_test, test_pred_rf)
r2_rf1 = r2_score(target_test, test_pred_rf)

# %%XGBoost

xgb_model1 = XGBRegressor(random_state=42)
best_xgb_random1 = tune_with_random_search(features1_train, target_train, xgb_model1)

pred = best_xgb_random1.predict(features1_test)

test_pred_xgb = pd.DataFrame(pred)
rmse1_xgb = root_mean_squared_error(target_test, test_pred_xgb)
r2_xgb1 = r2_score(target_test, test_pred_xgb)

# %% Print RMSEs 
print("For Non-Normalized Data, RSMEs:")
print(f"Linear Regression RMSE: {rmse1_lr:.2f}")
print(f"Random Forest RMSE: {rmse1_rf:.2f}")
print(f"Extreme Gradient Boosted RMSE: {rmse1_xgb:.2f}")

# =============================================================================
# # %%Normalization Function 
# def normalizer(data, scaler):
#         #Separate out features to not be normalized
#         data_norm = data[['dayOfYear','Hour','month_number','Production']]
#         data_nonorm = data[['Source','month_sinhalf', 'hour_sinhalf']]
# 
#         #Normalize and reappend
#         data_norm = scaler.transform(data_norm) # output is numpy array
#         data_nonormnp = data_nonorm.to_numpy()
#         targets = data_norm[:,3]
#         data_norm = np.delete(data_norm, 3, axis = 1)
#         features = np.concatenate((data_norm,data_nonormnp), axis = 1)
#         
#         return features, targets
#     
# =============================================================================
# %% Function for Inverse normalization to get value of Prediction
def output_inverter(output, scaler, fdim):

    k=np.expand_dims(output,1) # expand dim of array 
    op_copies = np.repeat(k, fdim, axis=1) # create proper number of copies for inverting
    op_inverted = scaler.inverse_transform(op_copies)[:,1] # 
    
    return op_inverted
# %% Testing Function
def tester(data_test, scaler, model, fdimmy):
    
    normdata = scaler.transform(data_test)
    #features_train, targets_train = normalizer(data_train, scaler)

    X = np.delete(normdata, 0, axis = 1)
    Y= normdata[:,0]
   # features_test, targets_test = normalizer(data_test, scaler)
   # targets = test_targets[6:]
    preds = output_inverter(model.predict(X), scaler, fdimmy)
    rsme = root_mean_squared_error(Y, preds)
    r_2 = r2_score(Y, preds)
                                   
    return rsme, r_2, preds
#%% Normalized Data

data_train , data_test = train_test_split(data, test_size=0.2, random_state=3)
#Prepare Training Data

scaler.fit(data_train)
normdata = scaler.transform(data_train)
#features_train, targets_train = normalizer(data_train, scaler)

X = np.delete(normdata, 1, axis = 1)
Y= normdata[:,1]
# %% Linear Regression

lr_model2 = LinearRegression()
lr_model2.fit(X,Y)

rmse2_lr, r2_lr2, preds_lr2 = tester(data_test, scaler, lr_model2, n_fdim)
# %% Random Forest

rf_model2 = RandomForestRegressor(random_state=42)
best_rf_grid2 = tune_with_grid_search(X, Y, rf_model2)

rmse2_rf, r2_rf2, preds_rf2 = tester(data_test, scaler, best_rf_grid2, n_fdim)
# %% XGBoost

xgb_model2 = XGBRegressor(random_state=42)
best_xgb_random2 = tune_with_random_search(X,Y, xgb_model2)

rmse2_xgb, r2_xgb2, preds_xgb2 = tester(data_test, scaler, best_xgb_random2, n_fdim)
# %%Print all RMSEs


print("For Normalized Data:")
print(f"Linear Regression RMSE: {rmse2_lr:.2f}")
print(f"Random Forest RMSE: {rmse2_rf:.2f}")
print(f"Extreme Gradient Boosted RMSE: {rmse2_xgb:.2f}")




# %% Save all models

pickle.dump(lr_model1 , open('C:/Users/mizaa/Desktop/Energy Forecasting/models/lr_1_wind.sav', 'wb'))
pickle.dump(best_rf_grid1 , open('C:/Users/mizaa/Desktop/Energy Forecasting/models/rf_1_wind.sav', 'wb'))
pickle.dump(best_xgb_random1 , open('C:/Users/mizaa/Desktop/Energy Forecasting/models/xgb_1_wind.sav', 'wb'))

pickle.dump(lr_model2, open('C:/Users/mizaa/Desktop/Energy Forecasting/models/lr_2_wind.sav', 'wb'))
pickle.dump(best_rf_grid2, open('C:/Users/mizaa/Desktop/Energy Forecasting/models/rf_2_wind.sav', 'wb'))
pickle.dump(best_xgb_random2, open('C:/Users/mizaa/Desktop/Energy Forecasting/models/xgb_2_wind.sav', 'wb'))