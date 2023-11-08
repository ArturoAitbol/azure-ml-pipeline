import argparse
import pandas as pd
import pickle
from sklearn.metrics import classification_report

# obtener parámetros:
parser = argparse.ArgumentParser("train")
parser.add_argument("--model", type=str, help="Trained model .pkl file")
parser.add_argument("--X_test", type=str, help="File of dependent variables for test")
parser.add_argument("--y_test", type=str, help="File of independent variable for test")
parser.add_argument("--report", type=str, help="Csv report showing the main classification metrics")
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

# Read X_test dataframe
X_test = pd.read_pickle(args.X_test)

# Read y_test dataframe
y_test = pd.read_pickle(args.y_test)

# Load model
filename = args.model
model = pickle.load(open(filename, "rb"))

# Make predictions on the test data
y_pred = model.predict(X_test)

# Replace y_true and y_pred with your actual true labels and predicted labels
output = classification_report(y_test, y_pred, target_names=["Not Potable", "Potable"], output_dict=True)
df_rep = pd.DataFrame(output).transpose()
# Display report in logs
print(df_rep)

# OUTPUT
# Save report in csv file
df_rep.to_csv(args.report)