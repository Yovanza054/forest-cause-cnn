import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from .config import CLASSES
from .synthetic import make_patch


def predict(model, x):
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(x)).argmax(1).numpy()


def plot_confusion(cm):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CLASSES)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Label sebenarnya")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig("outputs/confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_loss(history):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(range(1, len(history) + 1), history, color="#2e7d32")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Kurva Pelatihan")
    fig.tight_layout()
    fig.savefig("outputs/training_loss.png", dpi=150)
    plt.close(fig)


def cloud_test(model):
    rng = np.random.default_rng(7)
    lines = []
    for flag, name in ((False, "tanpa awan"), (True, "dengan awan")):
        xs, ys = [], []
        for idx, label in enumerate(CLASSES):
            for _ in range(100):
                xs.append(make_patch(label, rng, cloud=flag))
                ys.append(idx)
        p = predict(model, np.stack(xs))
        lines.append(f"akurasi {name}: {(p == np.array(ys)).mean():.3f}")
    print("\n".join(lines))
    with open("outputs/cloud_test.txt", "w") as f:
        f.write("\n".join(lines) + "\n")


def evaluate(model, x_te, y_te, history):
    pred = predict(model, x_te)
    report = classification_report(y_te, pred, target_names=CLASSES, digits=3)
    print(report)
    with open("outputs/report.txt", "w") as f:
        f.write(report)
    plot_confusion(confusion_matrix(y_te, pred))
    plot_loss(history)
    cloud_test(model)
