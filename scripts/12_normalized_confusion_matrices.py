import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

PREDICTION_PATH = "data/model_predictions.csv.gz"
FIGURE_DIR = "figures"

os.makedirs(FIGURE_DIR, exist_ok=True)

predictions = pd.read_csv(PREDICTION_PATH)

models = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest"
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (model_key, model_name) in zip(axes, models.items()):
    y_true = predictions["actual"]
    y_pred = predictions[f"{model_key}_prediction"]

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
        normalize="true"
    )

    image = ax.imshow(cm, interpolation="nearest")

    ax.set_title(model_name)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-Problematic", "Problematic"])
    ax.set_yticklabels(["Non-Problematic", "Problematic"])

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{cm[i, j]:.2%}",
                ha="center",
                va="center"
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle("Normalized Confusion Matrix Comparison")
fig.tight_layout()

output_path = os.path.join(
    FIGURE_DIR,
    "normalized_confusion_matrix_comparison.png"
)

plt.savefig(output_path, dpi=300)
plt.close()

print(f"Saved: {output_path}")
