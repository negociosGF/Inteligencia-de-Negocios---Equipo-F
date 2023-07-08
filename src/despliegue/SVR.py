
from sklearn.svm import SVR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


def app():
  st.title('Model - SVR')
  st.subheader
  plt.style.use('fivethirtyeight')

  """Cargamos los datos"""
  from pandas_datareader import data as pdr
  import yfinance as yfin
  yfin.pdr_override()

  st.subheader("Obtener datos de Yahoo finance")
  start = st.date_input('Start' , value=pd.to_datetime('2014-1-1'))
  end = st.date_input('End' , value=pd.to_datetime('2014-1-28'))
  user_input = st.text_input('Introducir cotización bursátil' , 'AMZN')

  df = pdr.get_data_yahoo(user_input, start, end)
  st.write(df)
  st.subheader("Detalles de los datos")
  #mostramos las filas y columnas
  st.write(df.shape)

  st.write(df.describe())

  #obtener la ultima fila 
  st.write("obtener la ultima fila")
  actual_price = df.tail(1)
  st.write(actual_price)

  #preparar datos de entrenamiento  ek modelo sCR
  #obtener todo los datos execpto la ultima fila
  st.write("obtener todo los datos execpto la ultima fila")
  df = df.head(len(df)-1)
  # print los nuevos datos
  st.write(df)

  #create una lista vacia independiente y dependiente

  days = list()
  adj_close_prices = list()

  #obtener la fecha y el precio de cieree ajustado
  df_days = df.index
  df_adj_close = df.loc[:, 'Adj Close'].to_numpy().astype(float)

  #create el conjunto de datos independiente
  for day in df_days:
    days.append([int(day.strftime('%d'))])
  # days = [df_days]
  adj_close_prices = df_adj_close

  # #create el conjunto de datos dependientes
  # for adj_close_price in df_adj_close:
  #   adj_close_prices.append(adj_close_price)

  #
  # st.write(days)
  # st.write(adj_close_prices)

  #creamos el support vector regression models
  # Create and train a SVR model using un kernel lineal

  lin_svr = SVR(kernel='linear', C=1000.0)
  lin_svr.fit(days, adj_close_prices)

  poly_svr = SVR(kernel='poly', C=1000.0, degree = 2)
  poly_svr.fit(days, adj_close_prices)

  rbf_svr = SVR(kernel='rbf', C=1000.0, gamma = 0.15)
  rbf_svr.fit(days, adj_close_prices)

  fig = plt.figure(figsize=(16,8))
  plt.scatter(days, adj_close_prices, color = 'red', label = 'Data')
  plt.plot(days, rbf_svr.predict(days), color = 'green', label='RBF Modelo')
  plt.plot(days, poly_svr.predict(days), color = 'orange', label='Polynomial Modelo')
  plt.plot(days, lin_svr.predict(days), color = 'Blue', label='Linear Modelo')
  plt.legend()
  st.pyplot(fig)

  day = [[30]]

  st.write('The RBF SVR prediction', rbf_svr.predict(day))
  st.write('The Lineal SVR prediction', lin_svr.predict(day))
  st.write('The Polynomial SVR prediction', poly_svr.predict(day))

##########PLANTILLA####################
  # Evaluación del modelo
  from sklearn import metrics
  import plotly.express as px
  st.title('Evaluación del RBF SVR prediction')
  ## Métricas
  MAE=metrics.mean_absolute_error(day, rbf_svr.predict(day))
  MSE=metrics.mean_squared_error(day, rbf_svr.predict(day))
  RMSE=np.sqrt(metrics.mean_squared_error(day, rbf_svr.predict(day)))
  
  metricas = {
      'metrica' : ['Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error'],
      'valor': [MAE, MSE, RMSE]
  }
  metricas = pd.DataFrame(metricas)  
  ### Gráfica de las métricas
  st.subheader('Métricas de rendimiento') 
  fig = px.bar(        
      metricas,
      x = "metrica",
      y = "valor",
      title = "Métricas del RBF SVR prediction",
      color="metrica"
  )
  st.plotly_chart(fig)