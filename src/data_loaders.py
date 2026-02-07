from pathlib import Path
from typing import Tuple, List
import random

from torchvision.datasets import ImageFolder
from torch.utils.data import ConcatDataset

from src.augmentations import get_transforms


ROOT = Path("data/prepared")

TRAIN_REAL = ROOT / "train_real"
TRAIN_SYN = ROOT / "train_synthetic"
TEST_FIXED = ROOT / "test_fixed"

VALID_EXPERIMENTS = {"E1", "E2", "E3", "E4"}
VALID_AUGS = {"A1", "A2", "A3", "A4"}


def _assert_exists(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"Brak folderu: {p.resolve()}")


def get_datasets(experiment: str, aug_variant: str, image_size: int = 224, seed: int = 42) -> Tuple[ImageFolder, ImageFolder]:
    """
    Zwraca (train_dataset, test_dataset) zgodnie z E1–E4.
    Test jest zawsze stały: data/prepared/test_fixed (REAL, 20 szt., 4/klasa).

    E1: train_real + train_synthetic
    E2: tylko train_synthetic
    E3: tylko train_real
    E4: wyrównane: bierzemy tylko 70 synthetic (przycięcie do rozmiaru real=70), deterministycznie (seed)
    """
    experiment = experiment.upper().strip()
    aug_variant = aug_variant.upper().strip()

    if experiment not in VALID_EXPERIMENTS:
        raise ValueError(f"Nieznany eksperyment: {experiment}. Użyj E1/E2/E3/E4.")
    if aug_variant not in VALID_AUGS:
        raise ValueError(f"Nieznana augmentacja: {aug_variant}. Użyj A1/A2/A3/A4.")

    _assert_exists(TRAIN_REAL)
    _assert_exists(TRAIN_SYN)
    _assert_exists(TEST_FIXED)

    train_tf, test_tf = get_transforms(aug_variant, image_size=image_size)

    # bazowe datasety (ImageFolder automatycznie tworzy mapę klas po nazwach folderów)
    ds_real = ImageFolder(str(TRAIN_REAL), transform=train_tf)
    ds_syn = ImageFolder(str(TRAIN_SYN), transform=train_tf)
    ds_test = ImageFolder(str(TEST_FIXED), transform=test_tf)

    if ds_real.classes != ds_syn.classes or ds_real.classes != ds_test.classes:
        raise RuntimeError(
            "Niezgodne klasy między folderami. "
            f"real={ds_real.classes}, syn={ds_syn.classes}, test={ds_test.classes}"
        )

    if experiment == "E3":
        return ds_real, ds_test

    if experiment == "E2":
        return ds_syn, ds_test

    if experiment == "E1":
        return ConcatDataset([ds_real, ds_syn]), ds_test  # type: ignore

    # E4: wyrównane — przycinamy synthetic do liczności real (70)
    # Uwaga: przycinamy po indeksach całego datasetu, deterministycznie.
    target = len(ds_real)
    if target <= 0:
        raise RuntimeError("train_real ma 0 próbek — nie powinno się zdarzyć.")

    if len(ds_syn) < target:
        raise RuntimeError(f"train_synthetic ma za mało próbek ({len(ds_syn)}) do wyrównania do {target}.")

    rng = random.Random(seed)
    indices = list(range(len(ds_syn)))
    rng.shuffle(indices)
    keep = indices[:target]

    # Zrobimy podzbiór przez ImageFolder + subset indices
    # (zwracamy dataset typu torch.utils.data.Subset)
    from torch.utils.data import Subset
    ds_syn_balanced = Subset(ds_syn, keep)

    return ds_syn_balanced, ds_test  # type: ignore
