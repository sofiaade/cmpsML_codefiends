import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from CODE.preprocess import load_data, preprocess_data, balance_classes
from CODE.models import train_models
from CODE.class_analysis import plot_class_distribution
from CODE.visualize import (
    plot_model_accuracies,
    plot_confusion_matrices,
    plot_f1_scores,
    plot_feature_correlation,
    plot_pca_scatter,
    plot_roc_curves
)
from CODE.utils import (
    save_models,
    export_metrics,
    export_predictions
)

from sklearn.model_selection import train_test_split

# Load and preprocess
df = load_data("INPUT/TRAIN/steps_tracker_dataset.csv")
plot_feature_correlation(df)  # before feature transformation
X, y = preprocess_data(df, label_column='mood')
plot_class_distribution(y)
X, y = balance_classes(X, y)
plot_pca_scatter(X, y)

# Split for ROC/predictions later
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train models
results = train_models(X, y)

# Print metrics
for name, result in results.items():
    print(f"\n=== {name} ===")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print("Classification Report:")
    for label, metrics in result['report'].items():
        if isinstance(metrics, dict):
            print(f"  {label}: {metrics}")

# Visualize model performance
plot_model_accuracies(results)
plot_f1_scores(results)
plot_confusion_matrices(results, class_names=sorted(y.unique()))
plot_roc_curves(results, X_test, y_test)

# Save models, predictions, metrics
save_models(results)
export_metrics(results)
export_predictions(results["Ensemble"]["model"], X_test, y_test)

