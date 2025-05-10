import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

def load_data(path):
    """
    Load CSV file into pandas DataFrame
    """
    df = pd.read_csv(path)
    return df

def engineer_features(df):
    """
    Create additional features based on domain logic
    """
    df = df.copy()
    # Prevent division by zero
    df['active_minutes'] = df['active_minutes'].replace(0, 1)
    df['distance_km'] = df['distance_km'].replace(0, 1)

    # Engineered features
    df['steps_per_minute'] = df['steps'] / df['active_minutes']
    df['calories_per_km'] = df['calories_burned'] / df['distance_km']
    df['hydration_index'] = df['water_intake_liters'] / df['distance_km']
    df['sleep_efficiency'] = df['sleep_hours'] / df['active_minutes']

    # Drop raw date column (non-numeric)
    if 'date' in df.columns:
        df.drop(columns=['date'], inplace=True)

    return df

def group_labels(y):
    """
    Combine similar moods into broader categories for more learnable labels
    """
    label_map = {
        'happy': 'positive',
        'energetic': 'positive',
        'sad': 'negative',
        'stressed': 'negative',
        'tired': 'neutral'
    }
    return y.map(label_map)

def preprocess_data(df, label_column='mood'):
    """
    Full preprocessing pipeline:
    - Feature engineering
    - Label grouping
    - Normalization
    - Return X and grouped y
    """
    df = engineer_features(df)
    y_raw = df[label_column]
    y_grouped = group_labels(y_raw)
    df = df.drop(columns=[label_column])

    # Keep only numeric features
    numeric_features = df.select_dtypes(include=['number'])

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(numeric_features)

    return pd.DataFrame(scaled_features, columns=numeric_features.columns), y_grouped

def balance_classes(X, y):
    """
    Balance dataset using SMOTE oversampling
    """
    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X, y)
    return X_bal, y_bal

