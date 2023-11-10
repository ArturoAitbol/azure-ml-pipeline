import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import os

# obtener parámetros:
parser = argparse.ArgumentParser("train")
parser.add_argument("--X_train", type=str, help="File of dependent variables for train")
parser.add_argument("--y_train", type=str, help="File of independent variable for train")



parser.add_argument("--model_folder", type=str, help="File of the logistic regression model trained")
args = parser.parse_args()

# Read X_train dataframe
X_train = pd.read_pickle(args.X_train)

# Read y_train dataframe
y_train = pd.read_pickle(args.y_train)

# Create a binary classification model (Logistic Regression in this case)
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# OUTPUTS
# Create output directory
folder = args.model_folder
if not os.path.isdir(folder):
    os.makedirs(folder)

# save model in a .pkl file
output_directory = os.path.join(folder, "model.pkl")
pickle.dump(model, open(output_directory, "wb"))