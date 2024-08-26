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

dataframe=pd.read_csv('../data/raw/bar_pass_prediction.csv')
dataframe["indxgrp"]=dataframe["indxgrp"].replace({'g 700+': 7, 'f 640-700': 6, 'e 580-640': 5, 'd 520-580': 4, 'c 460-520': 3, 'b 400-460': 2, 'a under 400': 1})
dataframe["indxgrp2"]=dataframe["indxgrp"].replace({'i 820+':8, 'g 700-760': 7, 'f 640-700': 6, 'e 580-640': 5, 'd 520-580': 4, 'c 460-520': 3, 'b 400-460': 2, 'a under 400': 1})
dataframe=dataframe.drop(["decile1b","decile3","decile1","index6040","zfygpa","age","DOB_yr","bar","bar1","bar2","bar1_yr","bar2_yr","bar_passed"],axis=1)
dataframe=dataframe.drop(["race","race2"],axis=1)
dataframe["race1"][dataframe["race1"].isnull()]="white"
dataframe=dataframe.drop(["sex"],axis=1)
dataframe["gender"][dataframe["gender"].isnull()]="female"
dataframe["male"][dataframe["male"].isnull()]=0
dataframe["grad"][dataframe["grad"].isnull()]="Y"
dataframe=dataframe.drop(["parttime"],axis=1)
dataframe["fulltime"]=dataframe["fulltime"].fillna(1.0)
dataframe=dataframe.drop(["cluster"],axis=1)
dataframe["tier"]=dataframe["tier"].fillna(np.random.choice([3.0,4.0,5.0,6.0,2.0,1.0], p=[0.358, 0.273, 0.175,0.092,0.076,0.026]),limit=34)
dataframe["tier"]=dataframe["tier"].fillna(np.random.choice([3.0,4.0,5.0,6.0,2.0,1.0], p=[0.358, 0.273, 0.175,0.092,0.076,0.026]),limit=26)
dataframe["tier"]=dataframe["tier"].fillna(np.random.choice([3.0,4.0,5.0,6.0,2.0,1.0], p=[0.358, 0.273, 0.175,0.092,0.076,0.026]),limit=17)
dataframe["tier"]=dataframe["tier"].fillna(np.random.choice([3.0,4.0,5.0,6.0,2.0,1.0], p=[0.358, 0.273, 0.175,0.092,0.076,0.026]),limit=9)
dataframe["tier"]=dataframe["tier"].fillna(np.random.choice([3.0,4.0,5.0,6.0,2.0,1.0], p=[0.358, 0.273, 0.175,0.092,0.076,0.026]),limit=7)
dataframe["tier"]=dataframe["tier"].fillna(np.random.choice([3.0,4.0,5.0,6.0,2.0,1.0], p=[0.358, 0.273, 0.175,0.092,0.076,0.026]),limit=3)
dataframe["fam_inc"]=dataframe["fam_inc"].fillna(3.0)
x = dataframe["gpa"].values #returns a numpy array
x=x.reshape(-1,1)
scaler = preprocessing.StandardScaler()
x_scaled =scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)
df=df.round(2)
df=pd.DataFrame(df.values,columns=["zgpa"])
dataframe["zgpa"]=dataframe["zgpa"].fillna(df["zgpa"])
df_blancos=pd.get_dummies(dataframe["race1"])
df_blancos["white"]=df_blancos["white"].astype(int)
dataframe_def=pd.concat([dataframe,df_blancos["white"]],axis=1)
dataframe_def=dataframe_def[['ID', 'lsat', 'ugpa', 'grad', 'zgpa', 'fulltime', 'fam_inc', 'gender',
       'male', 'race1', 'Dropout', 'white', 'other', 'asian', 'black', 'hisp',
       'pass_bar', 'tier', 'indxgrp', 'indxgrp2', 'dnn_bar_pass_prediction',
       'gpa']]
dataframe_def["white"]=dataframe_def["white"].astype("float64")
dataframe_def=dataframe_def.drop(["gender","race1"],axis=1)
dataframe_def["Dropout"]=(dataframe_def['Dropout'] == 'NO').astype(int)
dataframe_def.drop(7020)
dataframe_def.reindex()
dataframe_def["grad"]=(dataframe_def['grad'] == 'Y').astype(int)
dataframe_def=dataframe_def.drop(["indxgrp2"],axis=1)
dataframe_def=dataframe_def.drop(["ugpa"],axis=1)
dataframe_def=dataframe_def.drop(["dnn_bar_pass_prediction"],axis=1)
dataframe_def.to_csv("../data/processed/datos_procesados.csv",index=False)

dataframe_procesada=pd.read_csv("../data/processed/datos_procesados.csv",index_col=False)
seed=22
X_train_ryg, X_test_ryg, y_train_ryg, y_test_ryg = train_test_split(dataframe_procesada.drop(['pass_bar'], axis=1),dataframe_procesada['pass_bar'],test_size=0.25,random_state=seed)
X_train_ryg.to_csv("../data/train/datos_procesados_Xtrain_ryg.csv",index=False)
X_test_ryg.to_csv("../data/test/datos_procesados_Xtest_ryg.csv",index=False)
y_train_ryg.to_csv("../data/train/datos_procesados_ytrain_ryg.csv",index=False)
y_test_ryg.to_csv("../data/test/datos_procesados_ytest_ryg.csv",index=False)

ros = RandomOverSampler(random_state=seed)
X_ros, y_ros = ros.fit_resample(dataframe_procesada.drop(["pass_bar"],axis=1), dataframe_procesada['pass_bar'])
X_train_ros, X_test_ros, y_train_ros, y_test_ros = train_test_split(X_ros,y_ros,test_size=0.25,random_state=seed)
X_train_ros.to_csv("../data/train/datos_procesados_Xtrain_ros.csv",index=False)
X_test_ros.to_csv("../data/test/datos_procesados_Xtest_ros.csv",index=False)
y_train_ros.to_csv("../data/train/datos_procesados_ytrain_ros.csv",index=False)
y_test_ros.to_csv("../data/test/datos_procesados_ytest_ros.csv",index=False)

X_train_nocontrov=X_train_ryg.drop(["male","white","other","asian","black","hisp"],axis=1)
X_test_nocontrov=X_test_ryg.drop(["male","white","other","asian","black","hisp"],axis=1)
X_train_ros_nocontrov=X_train_ros.drop(["male","white","other","asian","black","hisp"],axis=1)
X_test_ros_nocontrov=X_test_ros.drop(["male","white","other","asian","black","hisp"],axis=1)
X_train_nocontrov.to_csv("../data/train/datos_procesados_X_train_nocontrov.csv",index=False)
X_test_nocontrov.to_csv("../data/test/datos_procesados_X_test_nocontrov.csv",index=False)
X_train_ros_nocontrov.to_csv("../data/train/datos_procesados_X_train_ros_nocontrov.csv",index=False)
X_test_ros_nocontrov.to_csv("../data/test/datos_procesados_X_test_ros_nocontrov.csv",index=False)