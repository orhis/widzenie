from pathlib import Path
import csv
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.models import get_model
from src.data_loaders import get_datasets
from src.train_utils import train_model, evaluate_model


def run_id(model_name: str, experiment: str, aug: str) -> str:
    return f"{model_name}__{experiment}__{aug}".replace("/", "_")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    models = ["efficientnet_b0", "shufflenet_v2_x1_0"]
    experiments = ["E1", "E2", "E3", "E4"]
    augs = ["A1", "A2", "A3", "A4"]

    # parametry treningu (możesz zmienić potem)
    epochs = 20 #######################################################
    lr = 1e-4
    batch_size = 128
    num_workers = 6
    seed = 42

    results_dir = Path("results")
    cm_dir = results_dir / "confusion_matrices"
    results_dir.mkdir(parents=True, exist_ok=True)
    cm_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "results.csv"

    rows = []
    total_runs = len(models) * len(experiments) * len(augs)
    run_counter = 0

    t_global = time.time()

    for m in models:
        for e in experiments:
            for a in augs:
                run_counter += 1
                rid = run_id(m, e, a)

                print("\n" + "=" * 90)
                print(f"[{run_counter}/{total_runs}] RUN: {rid}")
                print("=" * 90)

                # dane
                train_ds, test_ds = get_datasets(e, a, seed=seed)
                class_names = test_ds.classes

                train_loader = DataLoader(
                    train_ds, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=True,
                    persistent_workers=True
                )
                test_loader = DataLoader(
                    test_ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True,
                    persistent_workers=True
                )

                # model
                model = get_model(m, num_classes=len(class_names), pretrained=True).to(device)

                # trening
                train_time = train_model(model, train_loader, device, epochs=epochs, lr=lr)

                # ewaluacja
                metrics = evaluate_model(model, test_loader, device, num_classes=len(class_names))

                # zapisz confusion matrix
                cm_path = cm_dir / f"{rid}.csv"
                pd.DataFrame(metrics["confusion_matrix"], index=class_names, columns=class_names).to_csv(cm_path)

                row = {
                    "run_id": rid,
                    "model": m,
                    "experiment": e,
                    "augmentation": a,
                    "epochs": epochs,
                    "lr": lr,
                    "batch_size": batch_size,
                    "train_time_sec": train_time,
                    "test_accuracy": float(metrics["accuracy"]),
                    "test_macro_f1": float(metrics["macro_f1"]),
                    "confusion_matrix_csv": str(cm_path),
                }
                rows.append(row)

                # zrzut postępu do CSV po każdym runie (bezpieczne przy przerwaniu)
                pd.DataFrame(rows).to_csv(csv_path, index=False)

                print("accuracy:", row["test_accuracy"])
                print("macro_f1:", row["test_macro_f1"])
                print("saved cm:", cm_path)

    # finalny excel
    df = pd.DataFrame(rows)
    xlsx_path = results_dir / "results.xlsx"
    df.to_excel(xlsx_path, index=False)

    print("\n" + "=" * 90)
    print("DONE ALL RUNS")
    print("=" * 90)
    print("CSV :", csv_path.resolve())
    print("XLSX:", xlsx_path.resolve())
    print("CM  :", cm_dir.resolve())
    print("Total time (min):", (time.time() - t_global) / 60.0)


if __name__ == "__main__":
    main()
