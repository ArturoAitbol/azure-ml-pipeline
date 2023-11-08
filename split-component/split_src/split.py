import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

# obtener parámetros:
parser = argparse.ArgumentParser("train")
parser.add_argument("--dataset", type=str, help="Path to dataset")
parser.add_argument("--test_size", type=float, help="Split ratio for train")
parser.add_argument("--X_train", type=str, help="File of dependent variables for train")
parser.add_argument("--X_test", type=str, help="File of dependent variables for test")
parser.add_argument("--y_train", type=str, help="File of independent variable for train")
parser.add_argument("--y_test", type=str, help="File of independent variable for test")
args = parser.parse_args()

# #Lineas solo para verificar los argumentos. No necesarias en un ambiente de producción
# print("Hola desde split...")

# lines = [
#     f"Dataset: {args.dataset}",
#     f"Split ratio: {args.split_training_ratio}",
#     f"Train file: {args.data_train}",
#     f"Test file: {args.data_test}"
# ]
# print("Parametros: ...")
# # imprimir parámetros:
# for line in lines:
#     print(line)

# Read dataset
data = pd.read_csv(args.dataset)

# Dependent vs independent variables
X = data.drop(columns=['Potability'])
y = data['Potability']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)

# OUTPUTS
# Save dataframes in .pkl files
X_train.to_pickle(args.X_train)
X_test.to_pickle(args.X_test)
y_train.to_pickle(args.y_train)
y_test.to_pickle(args.y_test)

print(X_train)
print(X_test)
print(y_train)
print(y_test)
