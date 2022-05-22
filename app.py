import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st



start  = "2010-01-01"
end = "2022-12-31"

st.title("Stock Trend Prediction")

user_input = st.text_input("Enter Stock Name", "AAPL")
df = data.DataReader(user_input, 'yahoo', start, end)  # taking stocks of diff companies

# describing data
st.subheader("Data from 2010-2022")
st.write(df.describe())

# visualizations
st.subheader("Closing Price Vs Time Chart with 100 Moving Average")
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100,'r')
plt.plot(df.Close, 'b')
st.pyplot(fig)

st.subheader("Closing Price Vs Time Chart with 100 and 200 Moving Average")
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)



# splitting the data into training and testing

data_training = pd.DataFrame(df['Close'][0: int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

# scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

X_train = []
Y_train = []


# load my model
model = load_model("keras_model.h5")

# testing part
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

X_test = []
Y_test = []

for i in range(100, input_data.shape[0]):
    X_test.append(input_data[i-100: i])
    Y_test.append(input_data[i, 0])

X_test, Y_test  = np.array(X_test), np.array(Y_test)

# making predictions
Y_predicted = model.predict(X_test)

scaler = scaler.scale_

scale_factor = 1 / scaler[0]
Y_predicted = Y_predicted * scale_factor
Y_test = Y_test * scale_factor

st.subheader("Original Vs Predicted Price")
fig2 = plt.figure(figsize=(12, 6))
plt.plot(Y_test, 'b', label= "Original_Price")
plt.plot(Y_predicted, 'y', label= "Predicted_Price")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)


