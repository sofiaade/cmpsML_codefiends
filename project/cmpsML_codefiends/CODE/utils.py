import os
import pandas as pd
import joblib

def save_models(results, directory="MODEL/saved_models"):
    os.makedirs(directory, exist_ok=True)
    for name, result in results.items():
        model = result["model"]
        joblib.dump(model, os.path.join(directory, f"{name}.pkl"))

def export_metrics(results, save_path="OUTPUT/evaluation/metrics_summary.csv"):
    metrics = []
    for name, result in results.items():
        row = {
            "Model": name,
            "Accuracy": result["accuracy"],
            "F1 Score (Macro)": result["report"]["macro avg"]["f1-score"],
            "Precision (Macro)": result["report"]["macro avg"]["precision"],
            "Recall (Macro)": result["report"]["macro avg"]["recall"],
        }
        metrics.append(row)
    df = pd.DataFrame(metrics)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)

def export_predictions(model, X_test, y_test, save_path="OUTPUT/predictions.csv"):
    predictions = model.predict(X_test)
    df = pd.DataFrame({
        "True Label": y_test,
        "Predicted Label": predictions
    })
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)

