import argparse
import pandas as pd
import pickle
from sklearn.metrics import classification_report
import os

# obtener parámetros:
parser = argparse.ArgumentParser("eval")
parser.add_argument("--model_folder", type=str, help="Folder that stores the model .pkl file")
parser.add_argument("--X_test", type=str, help="File of dependent variables for test")
parser.add_argument("--y_test", type=str, help="File of independent variable for test")
parser.add_argument("--report_folder", type=str, help="Folder that contains a csv report showing the main classification metrics")
args = parser.parse_args()

# Read X_test dataframe
X_test = pd.read_pickle(args.X_test)

# Read y_test dataframe
y_test = pd.read_pickle(args.y_test)

# Load model
folder = args.model_folder
model_file = os.path.join(folder, "model.pkl")
# model_file = args.model
model = pickle.load(open(model_file, "rb"))

# Make predictions on the test data
y_pred = model.predict(X_test)

# Replace y_true and y_pred with your actual true labels and predicted labels
output = classification_report(y_test, y_pred, target_names=["Not Potable", "Potable"], output_dict=True)
df_rep = pd.DataFrame(output).transpose()
# Display report in logs
print(df_rep)

# OUTPUT
# Create output directory
folder = args.report_folder
if not os.path.isdir(folder):
    os.makedirs(folder)

output_directory = os.path.join(folder, "report.csv")

# Write csv file in output folder
df_rep.to_csv(output_directory)