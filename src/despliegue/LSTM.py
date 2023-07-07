# importamos las librerias a utilizar en el modelo
import streamlit as st
import math
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers
from pandas_datareader import data as pdr

# modelo
def app():

    st.title('LSTM')
    # se pueden definir aqui tmb las librerias


    # obtener la data de yahoo finance
    st.subheader("Obtener datos de Yahoo finance")
    # fechas de inicio y fin para la data
    start_date = st.date_input('Start Train' , value=pd.to_datetime('2021-01-01'))
    end_date = st.date_input('End Train' , value=pd.to_datetime('today'))
    # elegir la data
    user_input = st.text_input('Introducir cotización bursátil', 'ABX.TO')
    # obtenemos la data
    # df = pdr.get_data_yahoo(user_input, start_date, end_date)
    df = yf.download(user_input, start=start_date, end=end_date)

    # Miramos la data
    st.subheader("Detalles de los datos")
    st.write(df)


    # cierre
    close_prices = df['Adj Close']
    values = close_prices.values
    training_data_len = math.ceil(len(values)* 0.8)

    # escala
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(values.reshape(-1,1))
    train_data = scaled_data[0: training_data_len, :]

    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    test_data = scaled_data[training_data_len-60: , : ]
    x_test = []
    y_test = values[training_data_len:]

    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
    
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    model = keras.Sequential()
    model.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(layers.LSTM(100, return_sequences=False))
    model.add(layers.Dense(25))
    model.add(layers.Dense(1))
    model.summary()

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size= 1, epochs=3)

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    rmse = np.sqrt(np.mean(predictions - y_test)**2)
    rmse

    
    data = df.filter(['Adj Close'])
    train = data[:training_data_len]
    validation = data[training_data_len:]
    validation['Predictions'] = predictions

    # mostrar grafico
    st.subheader('Métricas de rendimiento')
    fig, ax = plt.subplots(figsize=(16,8))
    ax.set_title('Model')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price USD ($)')
    ax.plot(validation[['Adj Close', 'Predictions']])
    ax.legend(['Valor', 'Predictions', ''], loc='lower right')

    st.pyplot(fig)