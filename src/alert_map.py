import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from .config import CLASSES, PATCH_SIZE, SCENE_SIZE
from .synthetic import make_patch, make_intact_patch

COLORS = ["#f9a825", "#6d4c41", "#7cb342", "#d32f2f", "#1e88e5"]


def build_scene(rng):
    grid = SCENE_SIZE // PATCH_SIZE
    truth = np.full((grid, grid), -1)
    cells = [(r, c) for r in range(grid) for c in range(grid)]
    order = rng.permutation(len(cells))
    scene = np.zeros((4, SCENE_SIZE, SCENE_SIZE), dtype=np.float32)
    for k, i in enumerate(order):
        r, c = cells[i]
        if k < 10:
            idx = int(rng.integers(0, len(CLASSES)))
            patch = make_patch(CLASSES[idx], rng, cloud=False)
            truth[r, c] = idx
        else:
            patch = make_intact_patch(rng)
        scene[:, r * PATCH_SIZE:(r + 1) * PATCH_SIZE, c * PATCH_SIZE:(c + 1) * PATCH_SIZE] = patch
    return scene, truth


def detect_and_classify(model, scene, grid):
    pred = np.full((grid, grid), -1)
    model.eval()
    for r in range(grid):
        for c in range(grid):
            patch = scene[:, r * PATCH_SIZE:(r + 1) * PATCH_SIZE, c * PATCH_SIZE:(c + 1) * PATCH_SIZE]
            if (patch[0] - patch[1]).max() > 1.0:
                with torch.no_grad():
                    pred[r, c] = int(model(torch.from_numpy(patch[None])).argmax(1))
    return pred


def draw_alert_map(model):
    rng = np.random.default_rng(2024)
    scene, truth = build_scene(rng)
    pred = detect_and_classify(model, scene, truth.shape[0])
    cmap = ListedColormap(["#c8e6c9"] + COLORS)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.5))
    for ax, data, title in ((axes[0], truth, "Kebenaran (simulasi)"), (axes[1], pred, "Prediksi CNN")):
        ax.imshow(data + 1, cmap=cmap, vmin=0, vmax=len(COLORS))
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    handles = [Patch(color="#c8e6c9", label="hutan utuh")] + [
        Patch(color=COLORS[i], label=CLASSES[i]) for i in range(len(CLASSES))
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig("outputs/alert_map.png", dpi=150)
    plt.close(fig)
    valid = truth >= 0
    line = f"akurasi peta alert: {(truth[valid] == pred[valid]).mean():.3f} dari {valid.sum()} sel gangguan"
    print(line)
    with open("outputs/alert_map_accuracy.txt", "w") as f:
        f.write(line + "\n")
