# **Proyecto Final ML de Daniel Alejandro Salcedo Pérez**

El presente proyecto pretende crear seis modelos de clasificación que puedan predecir lo suficientemente bien. 

Se guardaran todos los modelos y los datos usados para su entrenamiento y evaluación posterior. Además, se han guardado diferentes versiones de todos estos (trained, untrained, raw, processed...).

+ Los datos, los cuales se encuentran en la carpeta *data* en formato csv, provienen originalmente de una [competición de Kaggle](https://www.kaggle.com/datasets/danofer/law-school-admissions-bar-passage/data "Title") de hace dos años. Estos datos fueron tomados previamente de un [estudio](/docs/LSAC%20National%20Longitudinal%20Bar%20Passage%20Study%20by%20Linda%20F.%20Wightman.pdf) de 1998 del LSAC (*Law School Admission Council*). El objetivo original de la competición era examinar la "imparcialidad" de los datos originales, los cuales agrupan a personas según características como etnia, raza, sexo, género, etc. 

    **Aunque la imparcialidad de los datos se haya mencionado no se ha realizado un examen en profundidad*.

+ Hay tres notebooks con el código necesario para el [procesamiento y análisis de los datos](/notebooks/EXPLORAR%20DATOS-LIMPIEZAEDA.ipynb), [el entrenamiento de modelos con diferentes versiones de estos datos, y la evaluación de las predicciones de los modelos](/notebooks/DIVISION%20TRAIN%20Y%20TEST-OVERSAMPLING-SELECCION%20MODELO.ipynb).

+ Los modelos guardados y sus hiperparametros se encuentran guardados en la carpeta *models*. Al igual que los datos, también los modelos se encuentran subdivididos. En este caso, la subdivisión se hace en subcarpetas (untrained,trained_w_data_controv...).

+ La carpeta *app* se encuentra vacia.

    **En construcción*

+ La carpeta *src* tiene exclusivamente el código necesario para los pasos indispensables para conseguir los resultados del ejercicio.

+ Por último, *requirements.txt* contiene las librerias usadas. Asímismo, se ha usado Python 3.10.14


Ejemplo de lo que se puede aquí:

+ ## Dataframe:
![image](images\intro.png)
+ ## Análisis:
![image](images\plot_1.png)
![image](images\plot_2.png)
![image](images\plot_3.png)
![image](images\plot_4.png)
![image](images\plot_5.png)
+ ## Modelos y evaluación:
![image](images\plot_6.png)
![image](images\plot_7.png)