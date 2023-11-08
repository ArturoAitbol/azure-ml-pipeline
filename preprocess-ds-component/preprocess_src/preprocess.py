import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("seaborn")

import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# obtener parámetros:
parser = argparse.ArgumentParser("train")
parser.add_argument("--dataset", type=str, help="Path to dataset")
parser.add_argument("--plot_style", type=str, help="Control the style of the pairplot")
parser.add_argument("--dataset_cleaned", type=str, help="Dataset cleaned")
parser.add_argument("--pair_plot", type=str, help="Name of the pair plot to save")

args = parser.parse_args()

print("Hola desde train...")
lines = [
    f"dataset: {args.dataset}",
    f"style: {args.plot_style}",
    f"cleaned: {args.dataset_cleaned}",
    f"pair_plot: {args.pair_plot}"
]
print("Parametros: ...")
# imprimir parámetros:
for line in lines:
    print(line)


# Read dataset
data = pd.read_csv(args.dataset)

# Check dimensions of the dataset
print(data.shape)

# Display number of null lines per column (just to log these values)
print(data.isnull().sum())

# Replace null values with the mean of each column
data = data.fillna(data.mean())

# Display number of null lines again (verify the process and displaying it in the logs)
print(data.isnull().sum())


# OUTPUTS:
# Save the cleaned dataset in a file
data.to_csv(args.dataset_cleaned)

# Create a pair plot
sns.set(style=args.plot_style)
sns_plot = sns.pairplot(data, diag_kind='kde')
plt.suptitle("Pair Plot of Columns")
# plt.show()
sns_plot.savefig(args.pair_plot)