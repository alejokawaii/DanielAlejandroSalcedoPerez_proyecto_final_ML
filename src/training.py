import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import imblearn
import pickle
import yaml
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import plot_tree
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
from imblearn.over_sampling import RandomOverSampler

X_train_ryg=pd.read_csv("../data/train/datos_procesados_Xtrain_ryg.csv",index=False)
X_test_ryg=pd.read_csv("../data/train/datos_procesados_Xtest_ryg.csv",index=False)
y_train_ryg=pd.read_csv("../data/train/datos_procesados_ytrain_ryg.csv",index=False)
y_test_ryg=pd.read_csv("../data/train/datos_procesados_ytest_ryg.csv",index=False)

X_train_nocontrov=pd.read_csv("../data/train/datos_procesados_X_train_nocontrov.csv",index=False)
X_test_nocontrov=pd.read_csv("../data/train/datos_procesados_X_test_nocontrov.csv",index=False)
X_train_ros_nocontrov=pd.read_csv("../data/train/datos_procesados_X_train_ros_nocontrov.csv",index=False)
X_test_ros_nocontrov=pd.read_csv("../data/train/datos_procesados_X_test_ros_nocontrov.csv",index=False)


X_train_ros=pd.read_csv("../data/train/datos_procesados_Xtrain_ros.csv",index=False)
X_test_ros=pd.read_csv("../data/test/datos_procesados_Xtest_ros.csv",index=False)
y_train_ros=pd.read_csv("../data/train/datos_procesados_ytrain_ros.csv",index=False)
y_test_ros=pd.read_csv("../data/test/datos_procesados_ytest_ros.csv",index=False)


tree_clf_puro = DecisionTreeClassifier()
tree_clf_puro=GridSearchCV(tree_clf_puro,"..\models\hyperparameters\hyperparameters_model_tree_clf.yaml",refit=True)
forest_clf_puro= RandomForestClassifier()
forest_clf_puro=GridSearchCV(forest_clf_puro,"..\models\hyperparameters\hyperparameters_model_forest_clf.yaml",refit=True)
reglog_clf_puro= LogisticRegression()
reglog_clf_puro=GridSearchCV(reglog_clf_puro,"..\models\hyperparameters\hyperparameters_model_reglog_clf.yaml",refit=True)
svc_clf_puro= SVC()
svc_clf_puro=GridSearchCV(svc_clf_puro,"..\models\hyperparameters\hyperparameters_model_svc_clf.yaml",refit=True)
knn_clf_puro= KNeighborsClassifier()
knn_clf_puro=GridSearchCV(knn_clf_puro,"..\models\hyperparameters\hyperparameters_model_knn_clf.yaml",refit=True)
km_clf_puro= KMeans()
km_clf_puro=GridSearchCV(km_clf_puro,"..\models\hyperparameters\hyperparameters_model_km_clf.yaml",refit=True)

respuesta_usuario=int(input("¿Datos Controversiales?"))
if respuesta_usuario == "Sí" or "Si" or "si" or "sI" or "sÍ":
    tree_clf_puro.fit(X_train_ryg, y_train_ryg)
    tree_clf=tree_clf_puro.best_estimator_
    forest_clf_puro.fit(X_train_ryg, y_train_ryg)
    forest_clf=forest_clf_puro.best_estimator_
    reglog_clf_puro.fit(X_train_ros,y_train_ros)
    reglog_clf=reglog_clf_puro.best_estimator_
    svc_clf_puro.fit(X_train_ros,y_train_ros)
    svc_clf=svc_clf_puro.best_estimator_
    knn_clf_puro.fit(X_train_ros,y_train_ros)
    knn_clf=knn_clf_puro.best_estimator_
    km_clf_puro.fit(X_train_ros,y_train_ros)
    km_clf=km_clf_puro.best_estimator_
if respuesta_usuario == "No" or "no" or "nO" or "NO":
    tree_clf_puro.fit(X_train_nocontrov, y_train_ryg)
    tree_clf=tree_clf_puro.best_estimator_
    forest_clf_puro.fit(X_train_nocontrov, y_train_ryg)
    forest_clf=forest_clf_puro.best_estimator_
    reglog_clf_puro.fit(X_train_ros_nocontrov,y_train_ros)
    reglog_clf=reglog_clf_puro.best_estimator_
    svc_clf_puro.fit(X_train_ros_nocontrov,y_train_ros)
    svc_clf=svc_clf_puro.best_estimator_
    knn_clf_puro.fit(X_train_ros_nocontrov,y_train_ros)
    knn_clf=knn_clf_puro.best_estimator_
    km_clf_puro.fit(X_train_ros_nocontrov,y_train_ros)
    km_clf=km_clf_puro.best_estimator_