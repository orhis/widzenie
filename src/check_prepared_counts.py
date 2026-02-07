from pathlib import Path

ROOT = Path("data/prepared")
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}

def count_images(p: Path):
    return sum(
        1 for f in p.rglob("*")
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    )

print("=" * 70)
print("CHECK: data/prepared")
print("=" * 70)

paths = {
    "test_fixed": ROOT / "test_fixed",
    "train_real": ROOT / "train_real",
    "train_synthetic": ROOT / "train_synthetic",
}

total = 0
for name, p in paths.items():
    n = count_images(p)
    total += n
    print(f"{name:<15} -> {n}")

print("-" * 70)
print(f"RAZEM           -> {total}")
print("=" * 70)
