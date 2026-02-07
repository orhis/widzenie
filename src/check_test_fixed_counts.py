from pathlib import Path

BASE = Path("dataset-h/Data/real/test_fixed")
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}

print("=" * 60)
print("LICZBA OBRAZÓW W test_fixed (PER RASA)")
print("=" * 60)

if not BASE.exists():
    print(f"❌ Folder nie istnieje: {BASE.resolve()}")
    raise SystemExit(1)

found_any = False

for rasa_dir in sorted(p for p in BASE.iterdir() if p.is_dir()):
    count = sum(
        1
        for f in rasa_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    )
    print(f"{rasa_dir.name:<6} -> {count}")
    found_any = True

if not found_any:
    print("❌ Brak podfolderów ras w test_fixed")

print("\nŚcieżka:", BASE.resolve())

