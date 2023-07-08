import yfinance as yf
import warnings
import streamlit as st

# Machine learning
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# For data manipulation
import pandas as pd
import numpy as np

# To plot
import matplotlib.pyplot as plt

def app():
    plt.style.use('seaborn-darkgrid')

    # To ignore warnings
    warnings.filterwarnings("ignore")

    ticker = st.text_input('Etiqueta de cotización', 'NFLX')
    st.write('La etiqueta de cotización actual es', ticker)

    tic = yf.Ticker(ticker)

    hist = tic.history(period="max", auto_adjust=True)

    df = hist

    # Crea variables predictoras
    df['Open-Close'] = df.Open - df.Close
    df['High-Low'] = df.High - df.Low

    # Guarda todas las variables predictoras en una variable X
    X = df[['Open-Close', 'High-Low']]

    # Variables objetivas
    y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

    split_percentage = 0.8
    split = int(split_percentage*len(df))

    # Train data set
    X_train = X[:split]
    y_train = y[:split]

    # Test data set
    X_test = X[split:]
    y_test = y[split:]

    # Support vector classifier
    cls = SVC().fit(X_train, y_train)

    df['Predicted_Signal'] = cls.predict(X)
    # Calcula los retornos diarios
    df['Return'] = df.Close.pct_change()
    # Calcula retornos de estrategia
    df['Strategy_Return'] = df.Return * df.Predicted_Signal.shift(1)
    # Calcula retornos acumulativos
    df['Cum_Ret'] = df['Return'].cumsum()

    st.write("Dataframe con retornos acumulativos")
    st.write(df)

    # Haz un plot de retornos de estrategia acumulativos
    df['Cum_Strategy'] = df['Strategy_Return'].cumsum()
    st.write("Dataframe con retornos de estrategia acumulativos")
    st.write(df)

    st.write("Plot Strategy Returns vs Original Returns")

    # Agrega sliders para seleccionar los rangos de datos
    start_index = st.slider('Índice de inicio', min_value=0, max_value=len(df)-1, value=0)
    end_index = st.slider('Índice de fin', min_value=0, max_value=len(df)-1, value=len(df)-1)

    # Ajusta los datos según los rangos seleccionados
    adjusted_df = df.iloc[start_index:end_index+1]

    fig = plt.figure()
    plt.plot(adjusted_df['Cum_Ret'], color='red', label='Retornos originales')
    plt.plot(adjusted_df['Cum_Strategy'], color='blue', label='Retornos de estrategia')
    plt.legend()
    st.pyplot(fig)
