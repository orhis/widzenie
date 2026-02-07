from pathlib import Path
import csv
import torch
from torch.utils.data import DataLoader

from src.models import get_model
from src.data_loaders import get_datasets
from src.train_utils import train_model, evaluate_model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    runs = [
        # (model, experiment, augmentation)
        ("efficientnet_b0", "E1", "A1"),
        ("efficientnet_b0", "E1", "A2"),
    ]

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "mini_results.csv"
    write_header = not csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "model", "experiment", "augmentation",
                "accuracy", "macro_f1", "train_time_sec"
            ])

        for model_name, experiment, aug in runs:
            print("\n" + "=" * 80)
            print(f"RUN: {model_name} | {experiment} | {aug}")
            print("=" * 80)

            train_ds, test_ds = get_datasets(experiment, aug)
            class_names = test_ds.classes

            train_loader = DataLoader(
                train_ds, batch_size=64, shuffle=True,
                num_workers=4, pin_memory=True
            )
            test_loader = DataLoader(
                test_ds, batch_size=64, shuffle=False,
                num_workers=4, pin_memory=True
            )

            model = get_model(model_name, num_classes=len(class_names), pretrained=True)
            model = model.to(device)

            train_time = train_model(
                model, train_loader, device,
                epochs=5, lr=1e-4
            )

            metrics = evaluate_model(
                model, test_loader, device, num_classes=len(class_names)
            )

            writer.writerow([
                model_name, experiment, aug,
                metrics["accuracy"], metrics["macro_f1"], train_time
            ])

            print("accuracy:", metrics["accuracy"])
            print("macro_f1:", metrics["macro_f1"])


if __name__ == "__main__":
    main()
