import streamlit as st
import pandas as pd
import streamlit.components.v1 as c
st.set_page_config(page_title="Modelo para ver si pasa el examen BAR de California",
                   page_icon=":BAR:")

st.write("Dedicada a Lucía. Te quiero.")

# Configuración de página
# Introducimos un selector
seleccion = st.sidebar.selectbox("Selecciona menu", ["Home", "EDA", "Predicciones"])

# Y en base a lo que diga el selector
if seleccion == "Home":

    # Indicamos título
    st.title("Modelo para ver si pasa el examen BAR de California")

    with st.expander("¿Qué es esta aplicación?"):
        st.write("""
                 Es una primera aproximación para predecir, en base a 
                 diferentes features (16) si una persona aprobaría o no
                 examen BAR de California.
                 """)

# Si hemos seleccionado EDA
elif seleccion == "EDA":

    st.write("Análisis exploratorio de datos")

    df = pd.read_csv("data\processed\datos_procesados.csv", sep=",")
    st.write(df.head())
    
# Si hemos seleccionado filtros
elif seleccion == "Filtros":

    st.write("Filtros aplicados a la selección")

    df = pd.read_csv("data\processed\datos_procesados.csv", sep=",")

