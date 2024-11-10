# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 21:29:40 2024

@author: jgcar
"""
###############################################################################
# Importar librerias
import math
import numpy as np
import pandas as pd
# Gráficos
import matplotlib.pyplot as plt
import seaborn as sns
# Metricas
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve

###############################################################################
# Funciones
def plot_confusion_matrix(y_true, y_pred, classes, normalize=False):
    '''
    Genera la matriz de confusión.

    Parameters
    ----------
    y_true : array
        Etiquetas reales.
    y_pred : array
        Etiquetas prdichas.
    classes : list
        Lista de valores/etiquetas de la variable objetivo.
    normalize : Bool, optional
        Muesta los valores en porcentaje.

    Returns
    -------
    None.
    '''
    try:
        # Calcula la matriz de confusion
        conf_mat = confusion_matrix(y_true, y_pred, labels=classes)
        if normalize:
            conf_mat  = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
        # Establece el area de trazado
        plt.figure(figsize=(8, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=classes)
        disp.plot(cmap=plt.cm.Blues)
        # Titulos y etiquetas
        plt.title('Matriz de Confusión')
        plt.xlabel('Etiqueta predicha')
        plt.ylabel('Etiqueta real')
        # Desactivar la grilla
        plt.grid(False)
        plt.tight_layout()
        plt.show()
    except Exception as ex:
        print("Se ha presentado una excepción", type(ex))

def plot_roc_curve(model, test_x, test_y, class_positive=1):
    '''
    Genera la gráfica de la curva roc.

    Parameters
    ----------
    model : model
        Modelo entrenado.
    test_X : array
        Conjunto de datos de pruebas, variables predictoras.
    test_y : TYPE
        Conjunto de datos de pruebas, variables objetivo.
    class_positive : int o str, optional
        Valor de la clase positiva.

    Returns
    -------
    None.
    '''
    # Calcula las predicciones de probabilidad
    try:
        y_score = model.predict_proba(test_x)[:, 1]
    except:
        y_score = model.decision_function(test_x)
    try:
        # Calcula las tasas
        fpr, tpr, _ = roc_curve(test_y, y_score, pos_label=class_positive)
        # Calcula el area bajo la curva
        roc_auc = auc(fpr, tpr)
        # Establece el area de trazado
        plt.figure(figsize=(8, 5))
        plt.plot(fpr, tpr, lw=1.5, label=f'ROC curve (area = {roc_auc:.2f})')
        # Genera la linea del 50%
        plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
        # Ajuste de limites
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        # Titulos y etiquetas
        plt.title('ROC curve')
        plt.legend(loc='lower right', fontsize=13)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.tight_layout()
        plt.show()
    except Exception as ex:
        print("Se ha presentado una excepción", type(ex))

