import streamlit as st
import pandas as pd
import streamlit.components.v1 as c
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
app = FastAPI()
aprobar_suspender = [0, 1]
class features(BaseModel):
    ID:float
    lsat:float
    grad:float
    zgpa:float
    fulltime:float
    fam_inc:float
    Dropout:float
    tier:float
    indxgrp:float
    gpa:float

def predict(features):
        """
        Usa los modelos entrenados para predecir

        Args:
            features (list): DataFrame de entrada con las columnas
                del Iris dataset.

        Returns:
            float: Predicción del modelo
        """
        model = joblib.load("./app/modelo_streamlit.joblib")
        df=pd.DataFrame(features,columns=["ID","lsat", "grad", "zgpa", "fulltime", "fam_inc", "Dropout", "tier", "indxgrp", "gpa"])
        return model.predict(df)


@app.get("/")
def read_root():
    return {"Message": "Esto es una API de un modelo (todavía en desarrollo) que permitirá predecir si un estudiante aprueba el examen BAR de California. Mirate el /docs, bro"}

@app.post("/predict")
def features(ID:float,lsat:float,grad:float,zgpa:float,fulltime:float,fam_inc:float,Dropout:float,tier:float,indxgrp:float,gpa:float):
    """_Resumen_

    Esto es una API de un modelo (todavía en desarrollo) que permitirá predecir si un estudiante aprueba el examen BAR de California.
    
    Args:
        ID: Any number / Cualquier número
        lsat: 
        grad: 
        zgpa: 
        fulltime: 
        fam_inc: 
        Dropout: 
        tier: 
        indxgrp: 
        gpa: 
    Returns:
        __type__:__description__
    """

    prediction=predict([[ID,lsat, grad, zgpa, fulltime, fam_inc, Dropout, tier, indxgrp, gpa]])
    return {"predicción": aprobar_suspender[int(prediction)]}