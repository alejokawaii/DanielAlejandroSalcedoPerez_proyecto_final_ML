import streamlit as st
import pandas as pd
import streamlit.components.v1 as c
from PIL import Image
import pickle
import sklearn
st.set_page_config(page_title="Modelo para ver si pasa el examen BAR de California",
                   page_icon=":BAR:")

#st.write("Dedicada a Lucía. Te quiero.")

# Configuración de página
# Introducimos un selector
seleccion = st.sidebar.selectbox("Selecciona menu", ["Home", "EDA", "Métricas del modelo" ,"Predicciones"])

# Y en base a lo que diga el selector
if seleccion == "Home":
    st.write("BIENVENIDO/A")
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
    st.write("Dataframe completo:")
    st.write(df)
    
    with st.expander("Datos en gráficas. Ejemplos:"):
        img2 = Image.open("images/plot_2.png")
        st.image(img2)
        img3 = Image.open("images/plot_3.png")
        st.image(img3)
        img4 = Image.open("images/plot_4.png")
        st.image(img4)
        img5 = Image.open("images/plot_5.png")
        st.image(img5)

    @st.cache_data
    def convert_df(df):
        return df.to_csv().encode("utf-8")
    csv = convert_df(df)

    st.download_button(label="Descargar csv de datos procesados", data=csv,file_name="datos_procesados.csv",mime="text/csv")
    st.page_link(page="https://www.kaggle.com/datasets/danofer/law-school-admissions-bar-passage/data", label="Fuente original de datos en bruto")

if seleccion == "Métricas del modelo":
    st.write("")
    img_regresion_log_7=Image.open("images/plot_7.png")
    st.image(img_regresion_log_7)
    img_regresion_log_8=Image.open("images/plot_8.png")
    st.image(img_regresion_log_8)
    img_regresion_log_9=Image.open("images/plot_9.png")
    st.image(img_regresion_log_9)

    st.write('"Importancia" de cada feature:')
    st.write("Index['ID', 'lsat', 'grad', 'zgpa', 'fulltime', 'fam_inc', 'male', 'Dropout','white', 'other', 'asian', 'black', 'hisp', 'tier', 'indxgrp', gpa]")
    st.write("array[0.   , 0.018, 0.001,  0.201,    0.011,     0.001,   0.001,     0.   ,  0.004,    0.   ,  0.001,      0.   ,  0.   ,0.012,   0.003,  0.001]")

# Si hemos seleccionado filtros
elif seleccion == "Predicciones":
    st.write("Predicciones")
    st.write("**El modelo ha sido entrenado con los datos procesados tras realizar un oversampling del target")


    def predict(features):
        """
        Usa los modelos entrenados para predecir

        Args:
            features (list): DataFrame de entrada con las columnas
                del Iris dataset.

        Returns:
            float: Predicción del modelo
        """
        model = pickle.load("models/trained_w_data_no_controv/untrained_model_reglog_cfl_w_data_no_controv.pkl")
        X_test_ros_nocontrov=pd.read_csv("data/test/datos_procesados_X_test_ros_nocontrov.csv")
        return model.predict(X_test_ros_nocontrov)

    aprobar_suspender = [0, 1]

    # Input fields
    st.header('')
    sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    sepal_width = st.number_input('Sepal Width (cm)', min_value=0.0, max_value=10.0, value=3.5, step=0.1)
    petal_length = st.number_input('Petal Length (cm)', min_value=0.0, max_value=10.0, value=1.5, step=0.1)
    petal_width = st.number_input('Petal Width (cm)', min_value=0.0, max_value=10.0, value=0.2, step=0.1)

    # Prediction button
    if st.button('Predict Iris Class'):
        # Make prediction
        features = [[sepal_length, sepal_width, petal_length, petal_width]]
        prediction = predict(features)

        # Display result
        st.header('Prediction')
        st.write()
        st.write(f'The iris flower is predicted to be: **{aprobar_suspender[int(prediction)]}**')
