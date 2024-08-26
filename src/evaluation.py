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

respuesta_usuario=int(input("¿Datos Controversiales?"))
if respuesta_usuario == "Sí" or "Si" or "si" or "sI" or "sÍ":
    tree_clf = pickle.load(open('../models/trained_w_data_controv/trained_model_tree_clf.pkl', 'rb'))
    forest_clf= pickle.load(open('../models/trained_w_data_controv/trained_model_forest_cfl.pkl', 'rb'))
    reglog_clf = pickle.load(open('../models/trained_w_data_controv/trained_model_reglog_cfl.pkl', 'rb'))
    svc_clf = pickle.load(open('../models/trained_w_data_controv/trained_model_svc_cfl.pkl', 'rb'))
    knn_clf = pickle.load(open('../models/trained_w_data_controv/trained_model_knn_cfl.pkl', 'rb'))
    km_clf = pickle.load(open('../models/trained_w_data_controv/trained_model_km_cfl.pkl', 'rb'))
    
    predicciones_tree=tree_clf.predict(X_test_ryg)
    predicciones_forest=forest_clf.predict(X_test_ryg)
    predicciones_reglog=reglog_clf.predict(X_test_ros)
    predicciones_svc=svc_clf.predict(X_test_ros)
    predicciones_knn=knn_clf.predict(X_test_ros)
    predicciones_km=km_clf.predict(X_test_ros)

    print(f"AUC: "+str(accuracy_score(predicciones_tree,y_test_ryg)))
    print(f"F1-Score:"+str(f1_score(predicciones_tree,y_test_ryg)))
    print(f"Precision Score: "+str(precision_score(predicciones_tree,y_test_ryg)))
    print(f"Recall Score: "+str(recall_score(predicciones_tree,y_test_ryg)))
    print(f"ROC-AUC Score: "+str(roc_auc_score(predicciones_tree,y_test_ryg)))
    print(f"ROC: "+str(roc_curve(predicciones_tree,y_test_ryg)))
    print(f"Precision Recall: "+str(precision_recall_curve(predicciones_tree,y_test_ryg)))
    print(f"Specifity: "+str(recall_score(predicciones_tree,y_test_ryg,pos_label=0)))
    sns.heatmap((confusion_matrix(predicciones_tree,y_test_ryg)),linewidths=1,annot=True,fmt="d")
    display = PrecisionRecallDisplay.from_estimator(
    tree_clf,X_test_ryg, y_test_ryg, name="Tree", plot_chance_level=True)
    _ = display.ax_.set_title("2-class Precision-Recall curve")
    fpr, tpr, thresholds = metrics.roc_curve(predicciones_tree,y_test_ryg)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                    estimator_name='example estimator')
    display.plot()
    plt.show()

    print(f"AUC: "+str(accuracy_score(predicciones_forest,y_test_ryg)))
    print(f"F1-Score:"+str(f1_score(predicciones_forest,y_test_ryg)))
    print(f"Precision Score: "+str(precision_score(predicciones_forest,y_test_ryg)))
    print(f"Recall Score: "+str(recall_score(predicciones_forest,y_test_ryg)))
    print(f"ROC-AUC Score: "+str(roc_auc_score(predicciones_forest,y_test_ryg)))
    print(f"ROC: "+str(roc_curve(predicciones_forest,y_test_ryg)))
    print(f"Precision Recall: "+str(precision_recall_curve(predicciones_forest,y_test_ryg)))
    print(f"Specifity: "+str(recall_score(predicciones_forest,y_test_ryg,pos_label=0)))
    sns.heatmap((confusion_matrix(predicciones_forest,y_test_ryg)),linewidths=1,annot=True,fmt="d")
    display = PrecisionRecallDisplay.from_estimator(
    forest_clf,X_test_ryg, y_test_ryg, name="Forest", plot_chance_level=True)
    _ = display.ax_.set_title("2-class Precision-Recall curve")

    print(f"AUC: "+str(accuracy_score(predicciones_reglog,y_test_ros)))
    print(f"F1-Score:"+str(f1_score(predicciones_reglog,y_test_ros)))
    print(f"Precision Score: "+str(precision_score(predicciones_reglog,y_test_ros)))
    print(f"Recall Score: "+str(recall_score(predicciones_reglog,y_test_ros)))
    print(f"ROC-AUC Score: "+str(roc_auc_score(predicciones_reglog,y_test_ros)))
    print(f"ROC: "+str(roc_curve(predicciones_reglog,y_test_ros)))
    print(f"Precision Recall: "+str(precision_recall_curve(predicciones_reglog,y_test_ros)))
    print(f"Specifity: "+str(recall_score(predicciones_reglog,y_test_ros,pos_label=0)))
    sns.heatmap((confusion_matrix(predicciones_reglog,y_test_ros)),linewidths=1,annot=True,fmt="d")
    fpr, tpr, thresholds = metrics.roc_curve(predicciones_forest,y_test_ryg)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                    estimator_name='example estimator')
    display.plot()
    plt.show()

    print(f"AUC: "+str(accuracy_score(predicciones_svc,y_test_ros)))
    print(f"F1-Score:"+str(f1_score(predicciones_svc,y_test_ros)))
    print(f"Precision Score: "+str(precision_score(predicciones_svc,y_test_ros)))
    print(f"Recall Score: "+str(recall_score(predicciones_svc,y_test_ros)))
    print(f"ROC-AUC Score: "+str(roc_auc_score(predicciones_svc,y_test_ros)))
    print(f"ROC: "+str(roc_curve(predicciones_svc,y_test_ros)))
    print(f"Precision Recall: "+str(precision_recall_curve(predicciones_svc,y_test_ros)))
    print(f"Specifity: "+str(recall_score(predicciones_svc,y_test_ros,pos_label=0)))
    sns.heatmap((confusion_matrix(predicciones_svc,y_test_ros)),linewidths=1,annot=True,fmt="d")
    display = PrecisionRecallDisplay.from_estimator(
    reglog_clf,X_test_ros, y_test_ros, name="Reg. Log.", plot_chance_level=True)
    _ = display.ax_.set_title("2-class Precision-Recall curve")
    fpr, tpr, thresholds = metrics.roc_curve(predicciones_reglog,y_test_ros)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                    estimator_name='example estimator')
    display.plot()
    plt.show()

    print(f"AUC: "+str(accuracy_score(predicciones_knn,y_test_ros)))
    print(f"F1-Score:"+str(f1_score(predicciones_knn,y_test_ros)))
    print(f"Precision Score: "+str(precision_score(predicciones_knn,y_test_ros)))
    print(f"Recall Score: "+str(recall_score(predicciones_knn,y_test_ros)))
    print(f"ROC-AUC Score: "+str(roc_auc_score(predicciones_knn,y_test_ros)))
    print(f"ROC: "+str(roc_curve(predicciones_knn,y_test_ros)))
    print(f"Precision Recall: "+str(precision_recall_curve(predicciones_knn,y_test_ros)))
    print(f"Specifity: "+str(recall_score(predicciones_knn,y_test_ros,pos_label=0)))
    sns.heatmap((confusion_matrix(predicciones_knn,y_test_ros)),linewidths=1,annot=True,fmt="d")
    display = PrecisionRecallDisplay.from_estimator(
    svc_clf,X_test_ros, y_test_ros, name="SVC", plot_chance_level=True)
    _ = display.ax_.set_title("2-class Precision-Recall curve")
    fpr, tpr, thresholds = metrics.roc_curve(predicciones_svc,y_test_ros)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                    estimator_name='example estimator')
    display.plot()
    plt.show()

    print(f"AUC: "+str(accuracy_score(predicciones_km,y_test_ros)))
    print(f"F1-Score:"+str(f1_score(predicciones_km,y_test_ros)))
    print(f"Precision Score: "+str(precision_score(predicciones_km,y_test_ros)))
    print(f"Recall Score: "+str(recall_score(predicciones_km,y_test_ros)))
    print(f"ROC-AUC Score: "+str(roc_auc_score(predicciones_km,y_test_ros)))
    print(f"ROC: "+str(roc_curve(predicciones_km,y_test_ros)))
    print(f"Precision Recall: "+str(precision_recall_curve(predicciones_km,y_test_ros)))
    print(f"Specifity: "+str(recall_score(predicciones_km,y_test_ros,pos_label=0)))
    sns.heatmap((confusion_matrix(predicciones_km,y_test_ros)),linewidths=1,annot=True,fmt="d")
    display = PrecisionRecallDisplay.from_estimator(
    knn_clf,X_test_ros, y_test_ros, name="KNN", plot_chance_level=True)
    _ = display.ax_.set_title("2-class Precision-Recall curve")
    fpr, tpr, thresholds = metrics.roc_curve(predicciones_knn,y_test_ros)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                    estimator_name='example estimator')
    display.plot()
    plt.show()

if respuesta_usuario == "No" or "no" or "nO" or "NO":
    tree_clf_sincontr = pickle.load(open('../models/trained_w_data_no_controv/trained_model_tree_clf_w_data_no_controv.pkl', 'rb'))
    forest_clf_sincontr= pickle.load(open('../models/trained_w_data_no_controv/trained_model_forest_cfl_w_data_no_controv.pkl', 'rb'))
    reglog_clf_sincontr = pickle.load(open('../models/trained_w_data_no_controv/trained_model_reglog_cfl_w_data_no_controv.pkl', 'rb'))
    svc_clf_sincontr = pickle.load(open('../models/trained_w_data_no_controv/trained_model_svc_cfl_w_data_no_controv.pkl', 'rb'))
    knn_clf_sincontr = pickle.load(open('../models/trained_w_data_no_controv/trained_model_knn_cfl_w_data_no_controv.pkl', 'rb'))
    km_clf_sincontr = pickle.load(open('../models/trained_w_data_no_controv/trained_model_km_cfl_w_data_no_controv.pkl', 'rb'))
    
    predicciones_tree_nocontr=tree_clf_sincontr.predict(X_test_nocontrov)
    predicciones_forest_nocontr=forest_clf_sincontr.predict(X_test_nocontrov)
    predicciones_reglog_nocontr=reglog_clf_sincontr.predict(X_test_ros_nocontrov)
    predicciones_svc_nocontr=svc_clf_sincontr.predict(X_test_ros_nocontrov)
    predicciones_knn_nocontr=knn_clf_sincontr.predict(X_test_ros_nocontrov)
    predicciones_km_nocontr=km_clf_sincontr.predict(X_test_ros_nocontrov)

    print(f"AUC: "+str(accuracy_score(predicciones_tree_nocontr,y_test_ryg)))
    print(f"F1-Score:"+str(f1_score(predicciones_tree_nocontr,y_test_ryg)))
    print(f"Precision Score: "+str(precision_score(predicciones_tree_nocontr,y_test_ryg)))
    print(f"Recall Score: "+str(recall_score(predicciones_tree_nocontr,y_test_ryg)))
    print(f"ROC-AUC Score: "+str(roc_auc_score(predicciones_tree_nocontr,y_test_ryg)))
    print(f"ROC: "+str(roc_curve(predicciones_tree_nocontr,y_test_ryg)))
    print(f"Precision Recall: "+str(precision_recall_curve(predicciones_tree_nocontr,y_test_ryg)))
    print(f"Specifity: "+str(recall_score(predicciones_tree_nocontr,y_test_ryg,pos_label=0)))
    sns.heatmap((confusion_matrix(predicciones_tree_nocontr,y_test_ryg)),linewidths=1,annot=True,fmt="d")
    display = PrecisionRecallDisplay.from_estimator(
    tree_clf_sincontr,X_test_nocontrov, y_test_ryg, name="KNN", plot_chance_level=True)
    _ = display.ax_.set_title("2-class Precision-Recall curve")
    fpr, tpr, thresholds = metrics.roc_curve(predicciones_tree_nocontr,y_test_ryg)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                    estimator_name='example estimator')
    display.plot()
    plt.show()

    print(f"AUC: "+str(accuracy_score(predicciones_forest_nocontr,y_test_ryg)))
    print(f"F1-Score:"+str(f1_score(predicciones_forest_nocontr,y_test_ryg)))
    print(f"Precision Score: "+str(precision_score(predicciones_forest_nocontr,y_test_ryg)))
    print(f"Recall Score: "+str(recall_score(predicciones_forest_nocontr,y_test_ryg)))
    print(f"ROC-AUC Score: "+str(roc_auc_score(predicciones_forest_nocontr,y_test_ryg)))
    print(f"ROC: "+str(roc_curve(predicciones_forest_nocontr,y_test_ryg)))
    print(f"Precision Recall: "+str(precision_recall_curve(predicciones_forest_nocontr,y_test_ryg)))
    print(f"Specifity: "+str(recall_score(predicciones_forest_nocontr,y_test_ryg,pos_label=0)))
    sns.heatmap((confusion_matrix(predicciones_forest_nocontr,y_test_ryg)),linewidths=1,annot=True,fmt="d")
    display = PrecisionRecallDisplay.from_estimator(
    forest_clf_sincontr,X_test_nocontrov, y_test_ryg, name="KNN", plot_chance_level=True)
    _ = display.ax_.set_title("2-class Precision-Recall curve")
    fpr, tpr, thresholds = metrics.roc_curve(predicciones_forest_nocontr,y_test_ryg)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                    estimator_name='example estimator')
    display.plot()
    plt.show()

    print(f"AUC: "+str(accuracy_score(predicciones_reglog_nocontr,y_test_ros)))
    print(f"F1-Score:"+str(f1_score(predicciones_reglog_nocontr,y_test_ros)))
    print(f"Precision Score: "+str(precision_score(predicciones_reglog_nocontr,y_test_ros)))
    print(f"Recall Score: "+str(recall_score(predicciones_reglog_nocontr,y_test_ros)))
    print(f"ROC-AUC Score: "+str(roc_auc_score(predicciones_reglog_nocontr,y_test_ros)))
    print(f"ROC: "+str(roc_curve(predicciones_reglog_nocontr,y_test_ros)))
    print(f"Precision Recall: "+str(precision_recall_curve(predicciones_reglog_nocontr,y_test_ros)))
    print(f"Specifity: "+str(recall_score(predicciones_reglog_nocontr,y_test_ros,pos_label=0)))
    sns.heatmap((confusion_matrix(predicciones_reglog_nocontr,y_test_ros)),linewidths=1,annot=True,fmt="d")
    display = PrecisionRecallDisplay.from_estimator(
    reglog_clf_sincontr,X_test_ros_nocontrov, y_test_ros, name="KNN", plot_chance_level=True)
    _ = display.ax_.set_title("2-class Precision-Recall curve")
    fpr, tpr, thresholds = metrics.roc_curve(predicciones_reglog_nocontr,y_test_ros)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                    estimator_name='example estimator')
    display.plot()
    plt.show()

    print(f"AUC: "+str(accuracy_score(predicciones_svc_nocontr,y_test_ros)))
    print(f"F1-Score:"+str(f1_score(predicciones_svc_nocontr,y_test_ros)))
    print(f"Precision Score: "+str(precision_score(predicciones_svc_nocontr,y_test_ros)))
    print(f"Recall Score: "+str(recall_score(predicciones_svc_nocontr,y_test_ros)))
    print(f"ROC-AUC Score: "+str(roc_auc_score(predicciones_svc_nocontr,y_test_ros)))
    print(f"ROC: "+str(roc_curve(predicciones_svc_nocontr,y_test_ros)))
    print(f"Precision Recall: "+str(precision_recall_curve(predicciones_svc_nocontr,y_test_ros)))
    print(f"Specifity: "+str(recall_score(predicciones_svc_nocontr,y_test_ros,pos_label=0)))
    sns.heatmap((confusion_matrix(predicciones_svc_nocontr,y_test_ros)),linewidths=1,annot=True,fmt="d")
    display = PrecisionRecallDisplay.from_estimator(
    svc_clf_sincontr,X_test_ros_nocontrov, y_test_ros, name="KNN", plot_chance_level=True)
    _ = display.ax_.set_title("2-class Precision-Recall curve")
    fpr, tpr, thresholds = metrics.roc_curve(predicciones_svc_nocontr,y_test_ros)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                    estimator_name='example estimator')
    display.plot()
    plt.show()

    print(f"AUC: "+str(accuracy_score(predicciones_knn_nocontr,y_test_ros)))
    print(f"F1-Score:"+str(f1_score(predicciones_knn_nocontr,y_test_ros)))
    print(f"Precision Score: "+str(precision_score(predicciones_knn_nocontr,y_test_ros)))
    print(f"Recall Score: "+str(recall_score(predicciones_knn_nocontr,y_test_ros)))
    print(f"ROC-AUC Score: "+str(roc_auc_score(predicciones_knn_nocontr,y_test_ros)))
    print(f"ROC: "+str(roc_curve(predicciones_knn_nocontr,y_test_ros)))
    print(f"Precision Recall: "+str(precision_recall_curve(predicciones_knn_nocontr,y_test_ros)))
    print(f"Specifity: "+str(recall_score(predicciones_knn_nocontr,y_test_ros,pos_label=0)))
    sns.heatmap((confusion_matrix(predicciones_knn_nocontr,y_test_ros)),linewidths=1,annot=True,fmt="d")
    display = PrecisionRecallDisplay.from_estimator(
    knn_clf_sincontr,X_test_ros_nocontrov, y_test_ros, name="KNN", plot_chance_level=True)
    _ = display.ax_.set_title("2-class Precision-Recall curve")
    fpr, tpr, thresholds = metrics.roc_curve(predicciones_knn_nocontr,y_test_ros)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                    estimator_name='example estimator')
    display.plot()
    plt.show()

    print(f"AUC: "+str(accuracy_score(predicciones_km_nocontr,y_test_ros)))
    print(f"F1-Score:"+str(f1_score(predicciones_km_nocontr,y_test_ros)))
    print(f"Precision Score: "+str(precision_score(predicciones_km_nocontr,y_test_ros)))
    print(f"Recall Score: "+str(recall_score(predicciones_km_nocontr,y_test_ros)))
    print(f"ROC-AUC Score: "+str(roc_auc_score(predicciones_km_nocontr,y_test_ros)))
    print(f"ROC: "+str(roc_curve(predicciones_km_nocontr,y_test_ros)))
    print(f"Precision Recall: "+str(precision_recall_curve(predicciones_km_nocontr,y_test_ros)))
    print(f"Specifity: "+str(recall_score(predicciones_km_nocontr,y_test_ros,pos_label=0)))
    sns.heatmap((confusion_matrix(predicciones_km_nocontr,y_test_ros)),linewidths=1,annot=True,fmt="d")
    display = PrecisionRecallDisplay.from_estimator(
    km_clf_sincontr,X_test_ros_nocontrov, y_test_ros, name="KNN", plot_chance_level=True)
    _ = display.ax_.set_title("2-class Precision-Recall curve")
    fpr, tpr, thresholds = metrics.roc_curve(predicciones_km_nocontr,y_test_ros)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                    estimator_name='example estimator')
    display.plot()
    plt.show()