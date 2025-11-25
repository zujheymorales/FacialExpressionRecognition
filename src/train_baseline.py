# src/train_baseline.py

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix

from data_loader import get_data_generators
from models_baseline import build_baseline_cnn

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# output directories (relative to src/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def plot_learning_curves(history, out_prefix="baseline"):
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    epochs = range(1, len(acc) + 1)

    # Accuracy
    plt.figure()
    plt.plot(epochs, acc, label="Train Acc")
    plt.plot(epochs, val_acc, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Baseline CNN Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(FIGURES_DIR, f"{out_prefix}_accuracy.png"))
    plt.close()

    # Loss
    plt.figure()
    plt.plot(epochs, loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Baseline CNN Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(FIGURES_DIR, f"{out_prefix}_loss.png"))
    plt.close()

def plot_confusion_matrix(cm, class_names, out_prefix="baseline"):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix - Baseline CNN",
    )

    # Rotate x-tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Write numbers inside the squares
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"{out_prefix}_confusion_matrix.png"))
    plt.close()

def main():
    # 1. Load data
    train_gen, val_gen = get_data_generators(
        data_dir=os.path.join(PROJECT_ROOT, "data", "train"),
        img_size=(48, 48),
        batch_size=64,
        val_split=0.2,
        augment=False,   # no augmentation for baseline
    )

    num_classes = train_gen.num_classes
    input_shape = train_gen.image_shape  # (48, 48, 1)

    # 2. Build model
    model = build_baseline_cnn(input_shape=input_shape, num_classes=num_classes)

    # 3. Callbacks
    checkpoint_path = os.path.join(MODELS_DIR, "baseline_cnn_best.keras")

    checkpoint_cb = ModelCheckpoint(
        checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )

    earlystop_cb = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    # 4. Train
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=30,  # you can lower this while testing
        callbacks=[checkpoint_cb, earlystop_cb],
    )

    # 5. Save final model and history
    final_model_path = os.path.join(MODELS_DIR, "baseline_cnn_final.keras")
    model.save(final_model_path)

    history_path = os.path.join(RESULTS_DIR, "baseline_history.json")
    with open(history_path, "w") as f:
        json.dump(history.history, f)

    # 6. Learning curves
    plot_learning_curves(history, out_prefix="baseline")

    # 7. Confusion matrix + classification report
    val_gen.reset()
    y_true = val_gen.classes
    class_indices = val_gen.class_indices
    idx_to_class = {v: k for k, v in class_indices.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    y_pred_probs = model.predict(val_gen)
    y_pred = np.argmax(y_pred_probs, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, out_prefix="baseline")

    report_str = classification_report(y_true, y_pred, target_names=class_names)
    report_path = os.path.join(RESULTS_DIR, "baseline_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report_str)

    print("=== Baseline CNN training complete ===")
    print(f"Model saved to: {final_model_path}")
    print(f"History saved to: {history_path}")
    print(f"Classification report saved to: {report_path}")

if __name__ == "__main__":
    main()
