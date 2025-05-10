import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_class_distribution(y, save_path="OUTPUT/plots/class_distribution.png"):
    """
    Plots and saves the class distribution of labels.
    """
    class_counts = y.value_counts()
    plt.figure(figsize=(8, 5))
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.title("Mood Class Distribution")
    plt.ylabel("Count")
    plt.xlabel("Mood")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

