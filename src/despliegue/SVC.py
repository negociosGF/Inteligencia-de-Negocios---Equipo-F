
# Machine learning
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Para manipulación de los datos
import pandas as pd
import numpy as np
import io

# Para gráficos
import matplotlib.pyplot as plt


# Para ignorar amenazas
import warnings
warnings.filterwarnings("ignore")


from google.colab import files
import matplotlib.pyplot as plt
# %matplotlib inline

def app():
    plt.style.use('seaborn-darkgrid')
    """Cargamos el archivo que usaremos en el análisis:"""
    uploaded = files.upload()

    """Leemos el archivo que hemos subido para extraer los datos y los mostramos en pantalla:"""
    df = pd.read_excel(io.BytesIO(uploaded['SCCO.xlsx']))
    print(df)

    """Los datos deben procesarse antes de su uso, de modo que la columna Date tomará la función de índice para hacerlo. Así que primero cambiamos la columna Date como índice...:"""

    df.index = pd.to_datetime(df['Date'])
    df

    """...y luego borramos la columna Date original:"""

    df = df.drop(['Date'], axis='columns')
    df

    """Las variables explicativas o independientes se utilizan para predecir la variable de respuesta de valor. La X es un conjunto de datos que contiene las variables que se utilizan para la predicción. La **X** consiste en variables como 'Open – Close' y 'High – Low'. Estos pueden entenderse como indicadores en función de los cuales el algoritmo predecirá la tendencia.

    Primero creamos las variables predictivas:
    """

    df['Open - Close'] = df.Open - df.Close
    df['High - Low'] = df.High - df.Low

    """Luego las colocamos en la variable X:"""

    X = df[['Open - Close', 'High - Low']]
    X.head()

    """La variable objetivo es el resultado que el modelo de aprendizaje automático predecirá en función de las variables explicativas. **Y** es un conjunto de datos objetivo que almacena la señal comercial correcta que el algoritmo de aprendizaje automático intentará predecir. Si el precio de mañana es mayor que el precio de hoy, entonces compraremos la acción en particular, de lo contrario no tendremos ninguna posición en el. Almacenaremos +1 para una señal de compra y 0 para una posición sin en y. Usaremos ***where()function*** de NumPy para hacer esto."""

    y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    y

    """Dividiremos los datos en conjuntos de datos de entrenamiento y prueba. Esto se hace para que podamos evaluar la efectividad del modelo en el conjunto de datos de prueba."""

    split_percentage = 0.8
    split = int(split_percentage*len(df))

    # Data set Entrenamiento
    X_train = X[:split]
    y_train = y[:split]

    # Data set Prueba
    X_test = X[split:]
    y_test = y[split:]

    """Usaremos la función SVC() de la biblioteca sklearn.svm.SVC para crear nuestro modelo clasificador utilizando el método fit() en el conjunto de datos de entrenamiento."""

    cls = SVC().fit(X_train, y_train)

    """Vamos a predecir la señal (comprar o vender) usando la función ***cls.predict()***:"""

    df['Predicted_Signal'] = cls.predict(X)

    """Calculamos devoluciones diarias:"""

    df['Return'] = df.Close.pct_change()

    """Ahora calculamos los retornos de estrategia:"""

    df['Strategy_Return'] = df.Return *df.Predicted_Signal.shift(1)

    """Y por último los rendimientos acumulados:"""

    df['Cum_Ret'] = df['Return'].cumsum()
    df

    df['Cum_Strategy'] = df['Strategy_Return'].cumsum()
    df

    """Con la información obtenida, hacemos una comparación de devoluciones de estrategia de trama vs. las devoluciones originales:"""

    # Commented out IPython magic to ensure Python compatibility.


    plt.plot(df['Cum_Ret'],color='orange')
    plt.plot(df['Cum_Strategy'],color='green')