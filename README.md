# Proyecto 2: Azure ML Pipeline

## pipeline.ipynb
Notebook en que se implementó el código para crear el pipeline utilizando los componentes creados:
* Crea el cluster en el que se ejecutará el pipeline
* Carga los componentes
* Define el pipeline (archivo de entrada, nodos, conexión de nodos y salidas del pipeline)
* Crea el job
* Descarga las salidas del pipeline

## preprocess-ds-component
Componente que reemplaza los valores nulos del dataset por los promedios y guarda el dataset limpio ademas del pair plot

## split-component
Separa las variables dependientes "x" de la variable independiente "y" y hace la división de los datosa partir del test_size definido

## train-logistic-component
Entrena un modelo de Logistic Regression utilizando los datos de training (X_train, y_train)

## train-tree-component
Entrena un Decision Tree utilizando los datos de training (X_train, y_train)

## eval-model-component
Obtiene los valores predecidos por el modelo y los compara a los valores reales (X_test, y_test) para obtener las metricas del modelo

## video.mp4
Video explicativo del proyecto

## pipeline_output
Directorio que contiene las salidas del pipeline
* pair_plot_output.- Pair plot en formato jpg
* logistic_model_output.- Modelo Logistic Regression entrenado y guardado en un archivo .pkl
* tree_model_output.- Modelo Decision Tree entrenado y guardado en un archivo .pkl
* logistic_report.- Reporte de métricas del modelo Logistic Regression en formato .csv
* tree_report.- Reporte de métricas del modelo Decision Tree en formato .csv