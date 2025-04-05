import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

file_path = "steps_tracker_dataset.csv"  # Ensure the file is in the working directory
df = pd.read_csv(file_path)

#If any missing columns 
print(df.head())
print (df.isna().sum())
print (df.dropna())
print (df.info())

#Convering the date column to the correct format
df['date'] = pd.to_datetime(df['date'], format="%d/%m/%Y")
print (df.info())

# Step 3: Handle Categorical Data (Encoding 'mood')
label_encoder = LabelEncoder()
df['mood_encoded'] = label_encoder.fit_transform(df['mood'])
df.drop(columns=['mood'], inplace=True)

# Step 4: Normalize Numerical Features
scaler = MinMaxScaler()
numerical_cols = ['steps', 'distance_km', 'calories_burned', 'active_minutes', 'sleep_hours', 'water_intake_liters']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Step 5: Save Preprocessed Data
df.to_excel("preprocessed_steps_tracker.xlsx", index=False)

# Step 6: Generate and Save Distribution Plots
sns.set(style="whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for i, col in enumerate(numerical_cols):
    sns.histplot(df[col], bins=30, kde=True, ax=axes[i // 3, i % 3])
    axes[i // 3, i % 3].set_title(f"Distribution of {col} (Normalized)")

plt.tight_layout()
plt.savefig("preprocessing_plots.png")
plt.show()

# Step 7: Display the first few rows of the cleaned dataset
print(df.head())

