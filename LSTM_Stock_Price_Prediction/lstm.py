#Recurrent Neural Network using LSTM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

file = os.path.abspath(__file__)
datadir = os.path.dirname(file)


# Importing the training set
dataset_train = pd.read_csv(datadir + '\Google_Stock_Price_Train.csv')
#only getting the column 'Open' on the dataset
training_set = dataset_train.iloc[:, 1:2].values



# Feature scaling
#we normalize the data to set them between 0 and 1
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []

for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])


X_train, y_train = np.array(X_train), np.array(y_train)

 

# Reshaping
#the last 1 is the number of indicators/predictors
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



# Building the RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation (0.2 for dropout is usually a good number)
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regulisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regulisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regulisation
#in the last LSTM layer, we don't need the return_sequences
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compile the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the training set
#at each iteration, 32 bacthes of stock prices go into the neural network and backpropagated
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)




# Making the prediction and visualizing the results
# Importing the training set
dataset_test = pd.read_csv(datadir + '\Google_Stock_Price_Test.csv')
#only getting the column 'Open' on the dataset
real_stock_price = dataset_test.iloc[:, 1:2].values


# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)

#note that the prediction for each day is done based on the values on 60 days before it
#therefore, after training, we only want the values of 60 days before of our target (test) values
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60 :].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
#reconvert the scaled data back to their original form
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()






















