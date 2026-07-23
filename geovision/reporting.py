from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


def save_confusion_matrix(matrix, class_names, path):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(14, 12))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(destination, dpi=180)
    plt.close()

