import argparse
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle
import os

# obtener parámetros:
parser = argparse.ArgumentParser("train")
parser.add_argument("--X_train", type=str, help="File of dependent variables for train")
parser.add_argument("--y_train", type=str, help="File of independent variable for train")
parser.add_argument("--criterion", type=str, help="The function to measure the quality of a split")
parser.add_argument("--min_samples_split", type=int, help="The minimum number of samples required to split an internal node")
parser.add_argument("--max_depth", type=int, help="The maximum depth of the tree")
parser.add_argument("--model_folder", type=str, help="File of the Decision Tree model trained")
args = parser.parse_args()

# Read X_train dataframe
X_train = pd.read_pickle(args.X_train)

# Read y_train dataframe
y_train = pd.read_pickle(args.y_train)

# Create a Decision Tree model
dt = DecisionTreeClassifier(criterion= args.criterion, min_samples_split= args.min_samples_split, max_depth=args.max_depth)

# Train the model on the training data
dt.fit(X_train, y_train)

# OUTPUTS
# Create output directory
folder = args.model_folder
if not os.path.isdir(folder):
    os.makedirs(folder)

# save model in a .pkl file
output_directory = os.path.join(folder, "model.pkl")
pickle.dump(dt, open(output_directory, "wb"))