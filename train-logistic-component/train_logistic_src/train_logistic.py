import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

# obtener parámetros:
parser = argparse.ArgumentParser("train")
parser.add_argument("--X_train", type=str, help="File of dependent variables for train")
parser.add_argument("--y_train", type=str, help="File of independent variable for train")
parser.add_argument("--logistic_model", type=str, help="File of the logistic regression model trained")
args = parser.parse_args()

# #Lineas solo para verificar los argumentos. No necesarias en un ambiente de producción
# print("Hola desde split...")

# lines = [
#     f"X_train: {args.X_train}",
#     f"Split ratio: {args.split_training_ratio}",
#     f"Train file: {args.data_train}",
#     f"Test file: {args.data_test}"
# ]
# print("Parametros: ...")
# # imprimir parámetros:
# for line in lines:
#     print(line)

# Read X_train dataframe
X_train = pd.read_pickle(args.X_train)

# Read y_train dataframe
y_train = pd.read_pickle(args.y_train)

# Create a binary classification model (Logistic Regression in this case)
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# OUTPUTS
# save model in a .pkl file
filename = args.logistic_model
pickle.dump(model, open(filename, "wb"))