import numpy as np
from .config import PATCH_SIZE, CLASSES, CLOUD_PROB

SIGNATURE = {
    "pertanian": (0.55, 2.5, 3.0),
    "jalan": (0.55, 2.5, 2.5),
    "tebang_selektif": (0.40, 1.5, 1.5),
    "tambang": (0.65, 3.5, 3.5),
    "alami": (0.50, 2.0, 1.5),
}


def forest_background(rng, size):
    ndvi_before = rng.normal(0.82, 0.05, (size, size))
    ndvi_after = ndvi_before + rng.normal(0, 0.02, (size, size))
    vv = rng.normal(-8.0, 1.2, (size, size))
    vh = rng.normal(-14.0, 1.2, (size, size))
    return np.stack([ndvi_before, ndvi_after, vv, vh])


def smooth(mask, iterations):
    m = mask.astype(float)
    for _ in range(iterations):
        m = (m + np.roll(m, 1, 0) + np.roll(m, -1, 0) + np.roll(m, 1, 1) + np.roll(m, -1, 1)) / 5
    return m


def disturbance_mask(label, rng, size):
    mask = np.zeros((size, size))
    cy, cx = rng.integers(10, size - 10, 2)
    yy, xx = np.ogrid[:size, :size]
    if label == "pertanian":
        h, w = rng.integers(10, 18, 2)
        y0 = np.clip(cy - h // 2, 0, size - h)
        x0 = np.clip(cx - w // 2, 0, size - w)
        mask[y0:y0 + h, x0:x0 + w] = 1
    elif label == "jalan":
        t = rng.integers(1, 3)
        if rng.random() < 0.5:
            y = rng.integers(8, size - 8)
            mask[y:y + t, :] = 1
        else:
            x = rng.integers(8, size - 8)
            mask[:, x:x + t] = 1
    elif label == "tebang_selektif":
        for _ in range(rng.integers(5, 10)):
            y, x = rng.integers(3, size - 3, 2)
            mask[y:y + 2, x:x + 2] = 1
    elif label == "tambang":
        r = rng.integers(6, 10)
        blob = (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2
        noise = smooth(rng.random((size, size)) > 0.5, 3)
        mask = (blob & (noise > 0.35)).astype(float)
    else:
        r = rng.integers(8, 14)
        blob = (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2
        noise = smooth(rng.random((size, size)) > 0.5, 3)
        mask = (blob & (noise > 0.4)).astype(float)
    return mask


def add_access_road(x, label, rng, size):
    if label in ("pertanian", "tambang") and rng.random() < 0.5:
        y = rng.integers(0, size)
        x[1, y, :] -= 0.5
        x[2, y, :] += 3.0
    return x


def add_cloud(x, rng, size):
    cy, cx = rng.integers(0, size, 2)
    r = rng.integers(8, 16)
    yy, xx = np.ogrid[:size, :size]
    cmask = (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2
    x[0][cmask] = rng.normal(0.1, 0.02)
    x[1][cmask] = rng.normal(0.1, 0.02)
    return x


def normalize(x):
    x[0] = (x[0] - 0.5) / 0.3
    x[1] = (x[1] - 0.5) / 0.3
    x[2] = (x[2] + 8) / 4
    x[3] = (x[3] + 14) / 4
    return x.astype(np.float32)


def make_patch(label, rng, cloud=None):
    size = PATCH_SIZE
    x = forest_background(rng, size)
    mask = disturbance_mask(label, rng, size)
    drop, vv, vh = SIGNATURE[label]
    drop += rng.normal(0, 0.15)
    vv += rng.normal(0, 1.5)
    vh += rng.normal(0, 1.5)
    hit = mask > 0.5
    x[1][hit] -= drop
    x[2][hit] += vv
    x[3][hit] += vh
    x = add_access_road(x, label, rng, size)
    use_cloud = rng.random() < CLOUD_PROB if cloud is None else cloud
    if use_cloud:
        x = add_cloud(x, rng, size)
    return normalize(x)


def make_intact_patch(rng):
    size = PATCH_SIZE
    return normalize(forest_background(rng, size))


def make_dataset(n_per_class, seed):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for idx, label in enumerate(CLASSES):
        for _ in range(n_per_class):
            xs.append(make_patch(label, rng))
            ys.append(idx)
    xs = np.stack(xs)
    ys = np.array(ys, dtype=np.int64)
    perm = rng.permutation(len(ys))
    return xs[perm], ys[perm]
