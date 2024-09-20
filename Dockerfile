# Imagen base
FROM python:3.10.14

# Creamos una carpeta
RUN mkdir /app

# La establecemos como directorio de trabajo
WORKDIR /app

# Añadimos el contenido de src a src
ADD ./app /app
ADD ./data /app
ADD ./requirements.txt /app

# Instalamos las dependencias de python
RUN pip install --no-cache-dir -r requirements.txt

# Ejecutamos el comando para lanzar la API
CMD ["streamlit", "run", "app.py"]

# Exponemos el puerto del backend
EXPOSE 8501