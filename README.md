# Proyecto 2: Azure ML Pipeline

## pipeline.ipynb
Notebook en que se implementó el código para crear el pipeline utilizando los componentes creados:
* Carga los componentes
* Define el pipeline (archivo de entrada, salidas del pipeline)
* Descarga las salidas del pipeline

## preprocess-ds-component
Componente que reemplza los valores nulos del dataset por los promedios y guarda el dataset limpio ademas del pair plot

## split-component
Divide el dataset en train y test

## train-logistic-component
Entrena un modelo de Logistic Regression

## train-tree-component
Entrena un Decision Tree

## eval-model-component
Obtiene los valores predecidos por el modelo y los compara a los valores reales para obtener las metricas del modelo