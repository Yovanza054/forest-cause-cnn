import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from .config import SEED, SAMPLES_PER_CLASS, BATCH_SIZE, LEARNING_RATE, EPOCHS
from .synthetic import make_dataset
from .model import ForestCauseCNN


def train():
    torch.manual_seed(SEED)
    x, y = make_dataset(SAMPLES_PER_CLASS, SEED)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, stratify=y, random_state=SEED)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr)),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    model = ForestCauseCNN()
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(yb)
        history.append(total / len(y_tr))
        print(f"epoch {epoch + 1:02d}/{EPOCHS} loss={history[-1]:.4f}")
    torch.save(model.state_dict(), "outputs/model.pt")
    return model, x_te, y_te, history
