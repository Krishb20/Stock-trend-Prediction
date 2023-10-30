import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from keras.models import load_model

# start_date = '2020-01-01'
# end_date = '2023-06-16'

st.title('Stock Trend Prediction using LSTM Model')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
start_date = st.text_input('Enter Start Date (Format: YYYY-MM-DD)', '2020-01-01')
end_date = st.text_input('Enter End Date (Format: YYYY-MM-DD)', '2023-06-16')
df = yf.download(user_input, start=start_date, end=end_date)

#Describing data
st.subheader('Data from ' + start_date + ' to ' + end_date)
st.write(df.describe())

#Visualizations
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'r', label = '100 days MA')
plt.plot(ma200, 'g', label = '200 days MA')
plt.plot(df.Close, 'b', label = 'Closing Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)


#Splitting training and testing data

data_training = pd.DataFrame(df['Close'][0 : int(len(df)*0.70)]) #70%
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70) : int(len(df))]) #30%

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
data_training_arr = scaler.fit_transform(data_training)

#Loading pretrained model
model = load_model('LSTM_Model.h5')

#Testing Part
past_100_days = data_training.tail(100) #For predicting first value for the test data
final_df = past_100_days._append(data_testing, ignore_index = True)
input_data = scaler.fit_transform(final_df)

x_test = [] 
y_test = [] 

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])  

x_test, y_test = np.array(x_test), np.array(y_test)

#Predicted Values

y_predicted = model.predict(x_test)
scale_factor = 1/scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#Final Trend Visualization

st.subheader('Predicted vs Original Closing Price')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Closing Price')
plt.plot(y_predicted, 'r', label = 'Predicted Closing Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

