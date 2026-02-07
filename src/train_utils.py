import time
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def train_model(
    model,
    train_loader,
    device,
    epochs: int,
    lr: float,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    t0 = time.time()

    for ep in range(1, epochs + 1):
        running = 0.0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running += loss.item()

        print(f"epoch {ep}/{epochs}  loss={running/len(train_loader):.4f}")

    return time.time() - t0


@torch.no_grad()
def evaluate_model(model, loader, device, num_classes: int) -> Dict[str, Any]:
    model.eval()

    all_y = []
    all_pred = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        pred = torch.argmax(logits, dim=1)

        all_y.extend(y.cpu().tolist())
        all_pred.extend(pred.cpu().tolist())

    acc = accuracy_score(all_y, all_pred)
    f1m = f1_score(all_y, all_pred, average="macro")
    cm = confusion_matrix(all_y, all_pred, labels=list(range(num_classes)))

    return {
        "accuracy": acc,
        "macro_f1": f1m,
        "confusion_matrix": cm,
    }
