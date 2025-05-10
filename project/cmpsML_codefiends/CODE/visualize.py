import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.preprocessing import label_binarize
from itertools import cycle

def plot_class_distribution(y, save_path="OUTPUT/plots/class_distribution.png"):
    class_counts = y.value_counts()
    plt.figure(figsize=(8, 5))
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.title("Mood Class Distribution")
    plt.ylabel("Count")
    plt.xlabel("Mood")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_model_accuracies(results, save_path="OUTPUT/plots/model_accuracy.png"):
    accuracies = {name: res["accuracy"] for name, res in results.items()}
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()))
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_f1_scores(results, save_path="OUTPUT/plots/model_f1_scores.png"):
    f1_scores = {
        name: res["report"]["macro avg"]["f1-score"]
        for name, res in results.items()
    }
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(f1_scores.keys()), y=list(f1_scores.values()))
    plt.title("Macro Avg F1-Score Comparison")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrices(results, class_names, output_dir="OUTPUT/plots"):
    os.makedirs(output_dir, exist_ok=True)
    for name, res in results.items():
        cm = res["confusion"]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap='Blues', xticks_rotation=45)
        plt.title(f"{name} Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{name.lower()}_confusion_matrix.png"))
        plt.close()

def plot_feature_correlation(df, save_path="OUTPUT/plots/feature_correlation.png"):
    numeric_df = df.select_dtypes(include=['number'])
    corr = numeric_df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_pca_scatter(X, y, save_path="OUTPUT/plots/pca_scatter.png"):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X)
    df_pca = pd.DataFrame(reduced, columns=["PC1", "PC2"])
    df_pca['label'] = y.values

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="label", palette="Set2")
    plt.title("PCA Projection of Features")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_roc_curves(results, X_test, y_test, save_path="OUTPUT/plots/roc_auc.png"):
    classes = sorted(list(set(y_test)))
    y_bin = label_binarize(y_test, classes=classes)

    plt.figure(figsize=(10, 8))

    for model_name in ["SVM", "ANN"]:
        model = results[model_name]["model"]
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
        else:
            continue

        for i, color in zip(range(len(classes)), cycle(["red", "blue", "green", "orange", "purple"])):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=2, label=f"{model_name} - {classes[i]} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves by Class")
    plt.legend(loc="lower right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

