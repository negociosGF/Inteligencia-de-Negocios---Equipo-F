# importamos las librerias a utilizar en el modelo
import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pylab as plt
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import requests
from html.parser import HTMLParser
import nlp_rake
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

# modelo
def app():

    st.title('Text Mining')
    # se pueden definir aqui tmb las librerias


    # elegir la data
    user_input = st.text_input('Introducir el artículo en español a analizar', 'Coldplay')
    # obtenemos la data
    url = f'https://es.wikipedia.org/wiki/{user_input}'


    # usando una biblioteca requests para obtener los datos
    text = requests.get(url).content.decode('utf-8')
    print(text[:1000])


    # st.subheader('HTML Parser')
    # El siguiente paso es convertir los datos en la forma adecuada para su procesamiento
    # Necesitamos convertir el texto sin el formato. 
    # Usar el objeto HTMLPArser y vamos a definir una función que nos saque el texto de los
    #tags menos de script y style. 

    class MyHTMLParser(HTMLParser):
        script = False
        res = ""
        def handle_starttag(self, tag, attrs):
            if tag.lower() in ["script","style"]:
                self.script = True
        def handle_endtag(self, tag):
            if tag.lower() in ["script","style"]:
                self.script = False
        def handle_data(self, data):
            if str.strip(data)=="" or self.script:
                return
            self.res += ' '+data.replace('[ edit ]','')

    parser = MyHTMLParser()
    parser.feed(text)
    text = parser.res
    # st.write(text[265:1265])


    st.subheader('Extracción de palabras claves y significativas del texto')
    # Podemos personalizar Rake. Por ejemplo podemos decir que la longitud mínima
    # de una palabra clave es de 5 caracteres, la frecuencia mínima de una palabra clave
    # sea de 3 y el número de palabras compuestas sea de 2. 

    
    # elegir la data
    max_palabras = st.text_input('Introducir la longitud maxima de una palabra compuesta', '2')
    min_frecuencias = st.text_input('Introducir la frecuencia minima de una palabra', '3')
    minimo_caracteres = st.text_input('introducir el numero minimo de caracteres de una palabra', '5')

    extractor = nlp_rake.Rake(max_words = int(max_palabras), 
                              min_freq = int(min_frecuencias), 
                              min_chars = int(minimo_caracteres))
    result = extractor.apply(text)
    st.write(result)


    # st.subheader("Gráfico de Barras")
    # visualización
    # def plot(pair_list):
    #    k,v = zip(*pair_list)
    #    plt.bar(range(len(k)),v)
    #    plt.xticks(range(len(k)),k,rotation='vertical')
    #    st.pyplot(plt)

    # plot(result)


    # nube de palabras
    def show_wordcloud(result):
        wc = WordCloud(background_color='white',width=800,height=600)
        plt.figure(figsize=(15,7))
        plt.imshow(wc.generate_from_frequencies({ k:v for k,v in result }))
        plt.axis('off')
        st.pyplot(plt)
    
    show_wordcloud(result)