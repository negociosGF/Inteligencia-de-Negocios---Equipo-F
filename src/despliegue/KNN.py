from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import yfinance as yf
plt.style.use('seaborn-darkgrid')
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import classification_report, confusion_matrix
import streamlit as st

def app():
    ticker1 = st.text_input('Etiqueta de cotización', 'INTC')
    st.write('La etiqueta de cotización actual es', ticker1)

    intc = yf.Ticker(ticker1)
    hist = intc.history(period="max", auto_adjust=True)
    hist.index = pd.to_datetime(hist.index)

    df = hist

    df['Open-Close'] = df.Open - df.Close
    df['High-Low'] = df.High - df.Low

    X = df[['Open-Close', 'High-Low']]
    X.head()

    y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

    split_percentage = 0.7
    split = int(split_percentage * len(df))

    X_train = X[:split]
    y_train = y[:split]

    X_test = X[split:]
    y_test = y[split:]

    knn = KNeighborsClassifier(n_neighbors=15)
    knn.fit(X_train, y_train)

    # Datos predecidos 
    st.write("Dataframe con los resultados predecidos")
    df['Predicted_Signal'] = knn.predict(X)
    st.write(df)

    # Precisión del modelo
    st.write("Precisión del modelo")
    st.write(accuracy_score(knn.predict(X_test), y_test))

    # Rangos de fecha
    start_date = pd.to_datetime(st.date_input('Fecha de inicio', value=df.index.min())).tz_localize(df.index.tz)
    end_date = pd.to_datetime(st.date_input('Fecha de fin', value=df.index.max())).tz_localize(df.index.tz)

    # Filtra los datos según los rangos de fecha seleccionados
    filtered_df = df.loc[start_date:end_date]

    # Gráfica de la tasa de error vs. valor de K
    tasa_error = []
    for i in range(1, 40):
        knn_g = KNeighborsClassifier(n_neighbors=i)
        knn_g.fit(X_train, y_train)
        pred_i = knn_g.predict(X_test)
        tasa_error.append(np.mean(pred_i != y_test))

    fig = plt.figure(figsize=(10, 6), dpi=250)
    plt.plot(range(1, 40), tasa_error, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
    plt.title('Tasa de Error vs. Valor de K')
    plt.xlabel('K')
    plt.ylabel('Tasa de Error')
    st.pyplot(fig)

    knn = KNeighborsClassifier(n_neighbors=19)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    st.write('CON K=19')
    st.write(classification_report(y_test, pred))
