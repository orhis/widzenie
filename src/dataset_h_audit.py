from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple
from PIL import Image
import sys

# =========================
# KONFIG
# =========================
ROOT = Path("dataset-h/Data") # folder obok src/
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
EXPECTED_SPLITS = ["real", "synthetic"]  # w dataset-h tak to wygląda


@dataclass
class FileIssue:
    path: Path
    issue: str


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def scan_split(split_root: Path) -> Tuple[Counter, Dict[str, Counter], List[FileIssue]]:
    """
    Zwraca:
    - format_counter: Counter(ext -> count)
    - per_class: dict[class_name] -> Counter({"images": n, "broken": m, "other_files": k})
    - issues: lista problemów (np. plik nie da się otworzyć)
    """
    format_counter = Counter()
    per_class: Dict[str, Counter] = defaultdict(Counter)
    issues: List[FileIssue] = []

    if not split_root.exists():
        issues.append(FileIssue(split_root, "Brak katalogu splitu"))
        return format_counter, per_class, issues

    # klasy = katalogi bezpośrednio pod splitem
    class_dirs = [d for d in split_root.iterdir() if d.is_dir()]
    if not class_dirs:
        issues.append(FileIssue(split_root, "Brak katalogów klas (pusto lub zła struktura)"))
        return format_counter, per_class, issues

    for class_dir in sorted(class_dirs, key=lambda x: x.name.lower()):
        class_name = class_dir.name

        for p in class_dir.rglob("*"):
            if p.is_file():
                ext = p.suffix.lower()

                if ext in IMAGE_EXTS:
                    per_class[class_name]["images"] += 1
                    format_counter[ext] += 1

                    # sprawdzenie czy obraz da się otworzyć
                    try:
                        with Image.open(p) as im:
                            im.verify()  # szybka walidacja nagłówka
                    except Exception as e:
                        per_class[class_name]["broken"] += 1
                        issues.append(FileIssue(p, f"Uszkodzony / nieczytelny obraz: {e.__class__.__name__}"))
                else:
                    per_class[class_name]["other_files"] += 1
                    issues.append(FileIssue(p, f"Nieobrazkowy plik w klasie (ext={ext or 'brak'})"))

    return format_counter, per_class, issues


def print_table(title: str, rows: List[Tuple[str, str]]):
    print("\n" + title)
    print("-" * len(title))
    for k, v in rows:
        print(f"{k:<35} {v}")


def main():
    print("=" * 90)
    print("AUDYT DATASETU: dataset-h")
    print("=" * 90)
    print("CWD:", Path.cwd())
    print("Python:", sys.version.split()[0])
    print("Root:", ROOT.resolve())
    print()

    if not ROOT.exists():
        print("❌ dataset-h nie istnieje w bieżącym katalogu projektu.")
        sys.exit(1)

    all_issues: List[FileIssue] = []
    overall_formats = Counter()
    totals = Counter()

    # sprawdzamy strukturę splitów
    for split in EXPECTED_SPLITS:
        split_root = ROOT / split
        fmt, per_class, issues = scan_split(split_root)
        all_issues.extend(issues)

        # sumy splitu
        split_total_images = sum(c["images"] for c in per_class.values())
        split_broken = sum(c["broken"] for c in per_class.values())
        split_other = sum(c["other_files"] for c in per_class.values())

        totals[f"{split}_images"] += split_total_images
        totals[f"{split}_broken"] += split_broken
        totals[f"{split}_other"] += split_other

        overall_formats.update(fmt)

        # raport splitu
        print_table(
            f"SPLIT: {split}",
            [
                ("Katalog", str(split_root.resolve())),
                ("Liczba klas", str(len(per_class))),
                ("Obrazy", str(split_total_images)),
                ("Uszkodzone obrazy", str(split_broken)),
                ("Inne pliki", str(split_other)),
            ],
        )

        # klasy
        class_rows = []
        for cls in sorted(per_class.keys()):
            c = per_class[cls]
            class_rows.append((cls, f"images={c['images']}  broken={c['broken']}  other={c['other_files']}"))
        print_table(f"Klasy w {split} (liczności)", class_rows)

        # formaty splitu
        fmt_rows = [(ext, str(cnt)) for ext, cnt in fmt.most_common()]
        print_table(f"Formaty w {split}", fmt_rows if fmt_rows else [("brak", "0")])

    # PODSUMOWANIE
    total_images = totals["real_images"] + totals["synthetic_images"]
    print("\n" + "=" * 90)
    print("PODSUMOWANIE CAŁOŚCI")
    print("=" * 90)

    print_table(
        "Liczności łącznie",
        [
            ("REAL - obrazy", str(totals["real_images"])),
            ("SYNTHETIC - obrazy", str(totals["synthetic_images"])),
            ("ŁĄCZNIE obrazów", str(total_images)),
            ("REAL - uszkodzone", str(totals["real_broken"])),
            ("SYNTHETIC - uszkodzone", str(totals["synthetic_broken"])),
            ("REAL - inne pliki", str(totals["real_other"])),
            ("SYNTHETIC - inne pliki", str(totals["synthetic_other"])),
        ],
    )

    print_table(
        "Formaty łącznie (real+synthetic)",
        [(ext, str(cnt)) for ext, cnt in overall_formats.most_common()] or [("brak", "0")],
    )

    # ZGODNOŚĆ Z WYMAGANIAMI (na obecnym etapie, bez stałego testu)
    print("\n" + "=" * 90)
    print("SPRAWDZENIE WYMAGAŃ (na podstawie dataset-h)")
    print("=" * 90)

    checks = []
    checks.append(("5 klas?", "Do sprawdzenia per split powyżej (powinno być 5 i 5)"))
    checks.append(("Łącznie 220 obrazów?", f"obecnie: {total_images} (wymagane: 220)"))
    checks.append(("Min. 70 real?", f"real: {totals['real_images']} (wymagane >= 70)"))
    checks.append(("Min. 50 synthetic?", f"synthetic: {totals['synthetic_images']} (wymagane >= 50)"))
    checks.append(("Stały test 20 real (4/klasa)?", "NIE wyodrębniono jeszcze — zrobimy w kolejnym kroku"))
    print_table("Checklist", checks)

    # ISSUES (jeśli są)
    print("\n" + "=" * 90)
    print("WYKRYTE PROBLEMY (jeśli są)")
    print("=" * 90)

    # pokaż tylko sensowną liczbę na ekranie, resztę można przekierować do pliku
    if not all_issues:
        print("✅ Brak problemów wykrytych przez audyt (formaty i otwieralność OK).")
    else:
        # grupuj po typie problemu
        by_issue = defaultdict(int)
        for it in all_issues:
            by_issue[it.issue] += 1

        print("Podsumowanie problemów:")
        for k, v in sorted(by_issue.items(), key=lambda x: -x[1]):
            print(f"  {v:>4}  {k}")

        print("\nPrzykładowe pierwsze 30 wpisów:")
        for it in all_issues[:30]:
            print(f"- {it.path} :: {it.issue}")

        print("\n(Jeśli chcesz pełną listę, uruchom z przekierowaniem do pliku — patrz instrukcja poniżej.)")


if __name__ == "__main__":
    main()
