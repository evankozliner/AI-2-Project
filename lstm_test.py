""" Followed blog posts at: 
1. http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/ and 
2. https://github.com/jaungiers/LSTM-Neural-Network-for-Time-Series-Prediction """

import numpy
import matplotlib.pyplot as plt
import pandas
import math
import os
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from BalanceSheetDataExtractor import BalanceSheetDataExtractor

def build_model(layers):
    model = Sequential()

    model.add(LSTM(
        input_shape=(1, 4),
        units=layers[1],
        return_sequences=True))

    model.add(LSTM(
        layers[2],
        return_sequences=False))

    model.add(Dense(
        units=layers[3]))
    model.add(Activation("linear"))

    model.compile(loss="mse", optimizer="rmsprop")
    return model


def main():
    extractor = BalanceSheetDataExtractor('AMZN', '2010-12-31')
    data = extractor.get_all_data()
    
    scaler1 = MinMaxScaler(feature_range=(0, 1))
    scaler2 = MinMaxScaler(feature_range=(0, 1))
    
    x = scaler1.fit_transform(data[['CashAndCashEquivalentsAtCarryingValue', 'Assets', 'LiabilitiesCurrent', 'amznopen']])
    y = scaler2.fit_transform(data['amznclose'].as_matrix().reshape(-1, 1))
    
    x_train, x_test = x[0:1000], x[1000:len(x)]
    y_train, y_test = y[0:1000], y[1000:len(y)]
    
    # reshape input to be [samples, time steps, features]
    x_train = numpy.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    x_test = numpy.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
    
    # create and fit the LSTM network
    model = build_model([1, 50, 200, 1])
    model.fit(x_train, y_train, epochs=20, batch_size=100, verbose=2, validation_split=0.05)
    
    # make predictions
    trainPredict = scaler2.inverse_transform(model.predict(x_train))
    testPredict = scaler2.inverse_transform(model.predict(x_test))
    trainY =scaler2.inverse_transform(y_train)
    testY = scaler2.inverse_transform(y_test)
    
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY, testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))
    
    plt.plot(numpy.reshape(scaler2.inverse_transform(y), (len(y),1)))
    
    trainPlot = numpy.empty_like(numpy.reshape(y, (len(y),1)))
    trainPlot[:, :] = numpy.nan
    trainPlot[:len(trainPredict), :] = numpy.reshape(trainPredict, (len(trainPredict),1))
    plt.plot(trainPlot)
    
    testPlot = numpy.empty_like(numpy.reshape(y, (len(y),1)))
    testPlot[:, :] = numpy.nan
    testPlot[len(trainPredict):, :] = numpy.reshape(testPredict, (len(testPredict),1))
    plt.plot(testPlot)
    
    plt.show()

if __name__ == "__main__":
    main()
