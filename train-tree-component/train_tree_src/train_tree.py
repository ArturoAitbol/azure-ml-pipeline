import argparse
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

# --X_train input/X_train.pkl --y_train input/y_train.pkl --criterion entropy --min_samples_split 3 --_depthint 4 --tree_model output/tree_model.pkl

# obtener parámetros:
parser = argparse.ArgumentParser("train")
parser.add_argument("--X_train", type=str, help="File of dependent variables for train")
parser.add_argument("--y_train", type=str, help="File of independent variable for train")
parser.add_argument("--criterion", type=str, help="The function to measure the quality of a split")
parser.add_argument("--min_samples_split", type=int, help="The minimum number of samples required to split an internal node")
parser.add_argument("--max_depth", type=int, help="The maximum depth of the tree")
parser.add_argument("--tree_model", type=str, help="File of the Decision Tree model trained")
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

# Create a Decision Tree model
dt = DecisionTreeClassifier(criterion= args.criterion, min_samples_split= args.min_samples_split, max_depth=args.max_depth)
# Train the model on the training data
dt.fit(X_train,y_train)

# OUTPUTS
# save model in a .pkl file
filename = args.tree_model
pickle.dump(dt, open(filename, "wb"))