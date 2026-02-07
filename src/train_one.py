from pathlib import Path
import time
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.models import get_model
from src.data_loaders import get_datasets


def evaluate(model, loader, device, class_names):
    model.eval()
    all_y = []
    all_pred = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            pred = torch.argmax(logits, dim=1)

            all_y.extend(y.cpu().tolist())
            all_pred.extend(pred.cpu().tolist())

    acc = accuracy_score(all_y, all_pred)
    f1m = f1_score(all_y, all_pred, average="macro")
    cm = confusion_matrix(all_y, all_pred, labels=list(range(len(class_names))))
    return acc, f1m, cm


def main():
    # ===== konfiguracja pojedynczego runu (sanity check) =====
    model_name = "shufflenet_v2_x1_0"
    experiment = "E1"
    aug = "A1"
    epochs = 5
    batch_size = 64
    lr = 1e-4
    seed = 42
    num_workers = 4
    image_size = 224

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dane
    train_ds, test_ds = get_datasets(experiment, aug, image_size=image_size, seed=seed)

    # class names (ImageFolder ma je w .classes, ConcatDataset już nie — więc bierzemy z test_ds)
    class_names = test_ds.classes

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    # model
    model = get_model(model_name, num_classes=len(class_names), pretrained=True)
    model = model.to(device)

    # optymalizacja
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # trening
    t0 = time.time()
    model.train()
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

    train_time = time.time() - t0

    # ewaluacja
    acc, f1m, cm = evaluate(model, test_loader, device, class_names)

    # zapis wyników
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": model_name,
        "experiment": experiment,
        "augmentation": aug,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "train_time_sec": train_time,
        "test_accuracy": acc,
        "test_macro_f1": f1m,
        "class_names": class_names,
        "confusion_matrix": cm.tolist(),
    }

    with open(out_dir / "sanity_run.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n=== SANITY RUN DONE ===")
    print("accuracy:", acc)
    print("macro_f1:", f1m)
    print("saved:", (out_dir / "sanity_run.json").resolve())


if __name__ == "__main__":
    main()
