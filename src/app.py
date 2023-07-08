import streamlit as st
from multiapp import MultiApp
from despliegue import KNN, home, LSTM, text_mining

app = MultiApp()
st.markdown("# Inteligencia de Negocios - Equipo F - Semestre 2023-I ")

# Add all your application here
app.add_app("Home", home.app)
app.add_app("LSTM", LSTM.app)
app.add_app("Text Mining", text_mining.app)
app.add_app("KNN", KNN.app)

# app.add_app("Nombre que aparece en el desplegable",<SOLO el nombre del archivo .py de la carpeta despliegue>.app)

# The main app
app.run()