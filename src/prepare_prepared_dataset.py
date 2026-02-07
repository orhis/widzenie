from pathlib import Path
import shutil

SRC_ROOT = Path("dataset-h/Data")
DST_ROOT = Path("data/prepared")

COPIES = [
    (SRC_ROOT / "real" / "test_fixed", DST_ROOT / "test_fixed"),
    (SRC_ROOT / "real" / "train_pool", DST_ROOT / "train_real"),
    (SRC_ROOT / "synthetic" / "train_pool", DST_ROOT / "train_synthetic"),
]

def safe_rmtree(p: Path):
    if p.exists():
        shutil.rmtree(p)

def copy_tree(src: Path, dst: Path):
    if not src.exists():
        raise FileNotFoundError(f"Brak źródła: {src.resolve()}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)

def main():
    print("=" * 80)
    print("PREPARE: data/prepared")
    print("=" * 80)
    print("Źródło:", SRC_ROOT.resolve())
    print("Cel   :", DST_ROOT.resolve())
    print()

    safe_rmtree(DST_ROOT)

    for src, dst in COPIES:
        print(f"Kopiuję:\n  {src.resolve()}\n  -> {dst.resolve()}\n")
        copy_tree(src, dst)

    print("✅ Gotowe.")

if __name__ == "__main__":
    main()
