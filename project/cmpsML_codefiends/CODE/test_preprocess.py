import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from CODE.preprocess import load_data, preprocess_data

# Load the dataset
df = load_data("INPUT/TRAIN/steps_tracker_dataset.csv")

# Show available columns
print(df.columns)

# Preprocess using correct label
X, y = preprocess_data(df, label_column='mood')

# Show preview
print("=== Preprocessed Features ===")
print(X.head())

print("\n=== Labels ===")
print(y.head())

