 import pandas as pd
 import numpy as np
 import matplotlib.pyplot as plt
 import seaborn as sns
 from sklearn.preprocessing import LabelEncoder, MinMaxScaler

file_path = "steps_tracker_dataset.csv"  # Ensure the file is in the working directory
df = pd.read_csv(file_path)

#preprcoessing the data

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

#feature engineering


    # Feature engineering
    df["steps_per_minute"] = df["steps"] / df["active_minutes"].replace(0, np.nan)
    df["sleep_to_activity_ratio"] = df["sleep_hours"] / df["active_minutes"].replace(0, np.nan)
    df["hydration_per_1000_steps"] = df["water_intake_liters"] / (df["steps"] / 1000).replace(0, np.nan)
    df["calories_per_km"] = df["calories_burned"] / df["distance_km"].replace(0, np.nan)

    # Rolling averages
    df = df.sort_values("date")
    df["steps_7day_avg"] = df["steps"].rolling(window=7, min_periods=1).mean()
    df["calories_7day_avg"] = df["calories_burned"].rolling(window=7, min_periods=1).mean()

    # Save to Excel
    df.to_excel("feature_engineering_steps_tracker.xlsx", index=False)

    plt.figure(figsize=(10, 6))
    sns.histplot(df["steps_per_minute"].dropna(), kde=True)
    plt.title("Steps per Active Minute")
    plt.savefig("feature_engineering_plots.png")
    plt.show()
    plt.close()

    ''' mood_mapping = {
        "stressed": 1,
        "sad": 2,
        "neutral": 3,
        "happy": 4,
        "excited": 5
    }   
    df["mood_numeric"] = df["mood"].map(mood_mapping)

    plt.figure(figsize=(10, 6))
    sns.boxplot(x="mood", y="hydration_per_1000_steps", data=df)
    plt.title("Hydration per 1000 Steps by Mood")
    plt.savefig("feature_engineering_plots.png")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    plt.close()  '''

    plt.figure(figsize=(10, 6))
    sns.lineplot(x="date", y="steps", data=df, label="Steps")
    sns.lineplot(x="date", y="steps_7day_avg", data=df, label="7-Day Avg Steps")
    plt.xticks(rotation=45)
    plt.title("Steps and 7-Day Average Over Time")
    plt.savefig("feature_engineering_plots.png")
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.heatmap(df[["steps", "active_minutes", "sleep_hours", "calories_burned", "water_intake_liters"]].corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap of Original Metrics")
    plt.savefig("feature_engineering_plots.png")
    plt.show()
    plt.close()


