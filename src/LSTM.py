# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 10:35:08 2024

@author: mizaa
"""
import os 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# %%
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, r2_score

# %% Create Sequences from raw data
def Sequencer(fdata, seq_length):

    #target = data['Production'].values'
    # Normalize features 
    #data=data.drop(columns='Source')
    # Create sequences and corresponding targets
    X, y = [], []
    for i in range(len(fdata)-seq_length):
        X.append(fdata[i:i+seq_length])
        y.append(fdata[i+seq_length,0])

    X, y = np.array(X), np.array(y)

    return X, y
# %%Build LSTM model

def build_lstm_model(input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(256, activation='relu', return_sequences=True, input_shape=input_shape))  
    model.add(tf.keras.layers.LSTM(128, activation='relu', return_sequences=True, input_shape=256))
    model.add(tf.keras.layers.LSTM(64, activation='relu', return_sequences=False, input_shape=128))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))    
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))   
    model.add(tf.keras.layers.Dense(1))  # Output layer (regression)
    model.summary()
    optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001
)
    model.compile(optimizer=optimizer, loss='mae')
    return model

# %%Train the model

def train_model(model, X_train, y_train, epochs, batch_size):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_split=0.2 ,   callbacks=[
               tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, mode='min')
               ]
                )
    return history


# %% Main function to run the process

if __name__ == "__main__":
    
    #Controlling Variables
    #csv_file_path = 'C:/Users/mizaa/Desktop/Energy Forecasting/data/finaldata_1.csv'  # Path to your CSV file
    csv_file_path = 'C:/Users/mizaa/Desktop/Energy Forecasting/data/finaldata_0.csv'
    sequence_length = 6  # Sequence length (look-back window)
    
    
    data = pd.read_csv(csv_file_path)
    #Split data
    data = data.to_numpy()
    data_train, data_test = train_test_split(data,test_size = 0.2, shuffle = False, random_state=32)
   
   # Training Norm
    scaler = StandardScaler()
    ndata = scaler.fit(data_train)
    ndata = scaler.transform(data_train)
    print(ndata.shape)

    # Sequencer
    X, Y = Sequencer(ndata, sequence_length)

    # Input shape is (sequence_length, num_features)
    input_shape = (X.shape[1], X.shape[2])
    
# %%Build and train LSTM model

    print("Training LSTM Model")
    lstm_model = build_lstm_model(input_shape)
    lstm_history = train_model(lstm_model, X, Y, epochs=50, batch_size=32)
# %% Observe Training Loss
import matplotlib.pyplot as plt 
plt.plot(lstm_history.history['loss'], label='Training Loss')
plt.plot(lstm_history.history['val_loss'], label='Validation Loss')
    
plt.show()

# %% Test Model 
    
#Generate Normalized Sequences
ndata_test = scaler.transform(data_test)

X_test, Y_test = Sequencer(ndata_test, sequence_length)
Yt = data_test[:,0]
Yt = Yt[6:]
#Get predictions and denormalize 
prediction = lstm_model.predict(X_test) #predictions 
prediction.reshape(prediction.shape[0])
prediction_copies = np.repeat(prediction, X.shape[2], axis=-1) #transform compatible****** shape ambiguity
y_pred = scaler.inverse_transform(prediction_copies)[:,0] #get Production values back from normalized ones

print(y_pred.shape)
print(Yt.shape)
   
#test
rmse = root_mean_squared_error(Yt, y_pred)
r2s = r2_score(Yt, y_pred)
print(rmse)
print(r2s)

 


# %% Save Model
lstm_model.save("C:/Users/mizaa/Desktop/Energy Forecasting/models/LSTM25612864_solar.keras")
