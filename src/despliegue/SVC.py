import yfinance as yf
import streamlit as st

# Machine learning
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# For data manipulation
import pandas as pd
import numpy as np

# To plot
import matplotlib.pyplot as plt

# Cargar datos históricos fuera de la función app()
ticker = 'NFLX'
hist = yf.download(ticker, period="max", auto_adjust=True)
hist.index = pd.to_datetime(hist.index)

def app():
    plt.style.use('seaborn-darkgrid')

    df = hist

    # Crea variables predictoras
    df['Open-Close'] = df.Open - df.Close
    df['High-Low'] = df.High - df.Low

    # Guarda todas las variables predictoras en una variable X
    X = df[['Open-Close', 'High-Low']]

    # Variables objetivas
    y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

    split_percentage = 0.8
    split = int(split_percentage * len(df))

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

    # Agrega sliders para seleccionar los rangos de fecha
    start_date = st.date_input('Fecha de inicio', value=df.index.min()).date()
    end_date = st.date_input('Fecha de fin', value=df.index.max()).date()

    # Filtra los datos según los rangos de fecha seleccionados
    filtered_df = df.loc[start_date:end_date]

    st.write("Dataframe con retornos acumulativos")
    st.write(filtered_df)

    # Haz un plot de retornos de estrategia acumulativos
    filtered_df['Cum_Strategy'] = filtered_df['Strategy_Return'].cumsum()

    st.write("Dataframe con retornos de estrategia acumulativos")
    st.write(filtered_df)

    st.write("Plot Strategy Returns vs Original Returns")
    fig = plt.figure()
    plt.plot(filtered_df['Cum_Ret'], color='red', label='Retornos originales')
    plt.plot(filtered_df['Cum_Strategy'], color='blue', label='Retornos de estrategia')
    plt.legend()
    plt.close(fig)  # Cerrar el objeto figura
    st.pyplot(fig)
