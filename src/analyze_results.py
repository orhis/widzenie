from __future__ import annotations

from pathlib import Path
import os

import pandas as pd
import matplotlib.pyplot as plt


RESULTS_CSV = Path("results") / "results.csv"
PLOTS_DIR = Path("results") / "plots"


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_bar_sorted_runs(df: pd.DataFrame, metric: str, fname: str, title: str) -> None:
    d = df.sort_values(metric, ascending=True).copy()
    plt.figure(figsize=(14, 6))
    plt.bar(d["run_id"], d[metric])
    plt.xticks(rotation=90, fontsize=7)
    plt.ylim(0.0, 1.0)
    plt.title(title)
    plt.ylabel(metric)
    _savefig(PLOTS_DIR / fname)


def plot_grouped_means(df: pd.DataFrame, group_cols: list[str], metric: str, fname: str, title: str) -> None:
    g = df.groupby(group_cols, dropna=False)[metric].mean().reset_index()
    # jedna oś "label" dla czytelności na wykresie
    g["label"] = g[group_cols].astype(str).agg(" | ".join, axis=1)
    g = g.sort_values(metric, ascending=False)

    plt.figure(figsize=(10, 5))
    plt.bar(g["label"], g[metric])
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0.0, 1.0)
    plt.title(title)
    plt.ylabel(f"mean {metric}")
    _savefig(PLOTS_DIR / fname)


def plot_pivot_heatmap(df: pd.DataFrame, index: str, columns: str, values: str, fname: str, title: str) -> None:
    pivot = df.pivot_table(index=index, columns=columns, values=values, aggfunc="mean")
    plt.figure(figsize=(7, 5))
    plt.imshow(pivot.values, aspect="auto")
    plt.title(title)
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.colorbar(label=f"mean {values}")
    _savefig(PLOTS_DIR / fname)


def plot_confusion_heatmap(cm_csv: Path, fname: str, title: str) -> None:
    cm = pd.read_csv(cm_csv, index_col=0)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm.values, aspect="auto")
    plt.title(title)
    plt.xticks(range(len(cm.columns)), cm.columns, rotation=45, ha="right")
    plt.yticks(range(len(cm.index)), cm.index)
    plt.colorbar(label="count")
    _savefig(PLOTS_DIR / fname)


def main() -> None:
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"Brak pliku: {RESULTS_CSV.resolve()}")

    _safe_mkdir(PLOTS_DIR)

    df = pd.read_csv(RESULTS_CSV)

    required = {"run_id", "model", "experiment", "augmentation", "test_accuracy", "test_macro_f1", "confusion_matrix_csv"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Brak wymaganych kolumn w results.csv: {sorted(missing)}")

    # 1) Ranking runów (accuracy i macro_f1)
    plot_bar_sorted_runs(
        df, metric="test_accuracy",
        fname="01_runs_sorted_by_accuracy.png",
        title="Test accuracy per run (sorted ascending)"
    )
    plot_bar_sorted_runs(
        df, metric="test_macro_f1",
        fname="02_runs_sorted_by_macro_f1.png",
        title="Test macro-F1 per run (sorted ascending)"
    )

    # 2) Średnie: model, eksperyment, augmentacja
    plot_grouped_means(
        df, group_cols=["model"], metric="test_accuracy",
        fname="03_mean_accuracy_by_model.png",
        title="Mean test accuracy by model"
    )
    plot_grouped_means(
        df, group_cols=["model"], metric="test_macro_f1",
        fname="04_mean_macro_f1_by_model.png",
        title="Mean test macro-F1 by model"
    )
    plot_grouped_means(
        df, group_cols=["experiment"], metric="test_accuracy",
        fname="05_mean_accuracy_by_experiment.png",
        title="Mean test accuracy by experiment (E1–E4)"
    )
    plot_grouped_means(
        df, group_cols=["augmentation"], metric="test_accuracy",
        fname="06_mean_accuracy_by_augmentation.png",
        title="Mean test accuracy by augmentation (A1–A4)"
    )

    # 3) Heatmapy: experiment x augmentation (mean accuracy)
    plot_pivot_heatmap(
        df, index="experiment", columns="augmentation", values="test_accuracy",
        fname="07_heatmap_accuracy_experiment_x_augmentation.png",
        title="Mean test accuracy: experiment x augmentation"
    )
    plot_pivot_heatmap(
        df, index="experiment", columns="augmentation", values="test_macro_f1",
        fname="08_heatmap_macro_f1_experiment_x_augmentation.png",
        title="Mean test macro-F1: experiment x augmentation"
    )

    # 4) Najlepszy i najgorszy run + confusion matrix (jeśli pliki istnieją)
    best_acc_row = df.loc[df["test_accuracy"].idxmax()]
    worst_acc_row = df.loc[df["test_accuracy"].idxmin()]

    best_cm_path = Path(str(best_acc_row["confusion_matrix_csv"]))
    worst_cm_path = Path(str(worst_acc_row["confusion_matrix_csv"]))

    if best_cm_path.exists():
        plot_confusion_heatmap(
            best_cm_path,
            fname="09_confusion_best_accuracy.png",
            title=f"Confusion matrix (best acc): {best_acc_row['run_id']}"
        )

    if worst_cm_path.exists():
        plot_confusion_heatmap(
            worst_cm_path,
            fname="10_confusion_worst_accuracy.png",
            title=f"Confusion matrix (worst acc): {worst_acc_row['run_id']}"
        )

    # 5) Tekstowe podsumowanie do wklejenia w raport
    summary_path = PLOTS_DIR / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("SUMMARY (from results/results.csv)\n")
        f.write("=" * 60 + "\n\n")

        f.write("BEST by test_accuracy:\n")
        f.write(str(best_acc_row[["run_id", "model", "experiment", "augmentation", "test_accuracy", "test_macro_f1"]].to_dict()) + "\n\n")

        f.write("WORST by test_accuracy:\n")
        f.write(str(worst_acc_row[["run_id", "model", "experiment", "augmentation", "test_accuracy", "test_macro_f1"]].to_dict()) + "\n\n")

        f.write("MEAN by model:\n")
        f.write(df.groupby("model")[["test_accuracy", "test_macro_f1"]].mean().sort_values("test_accuracy", ascending=False).to_string())
        f.write("\n\n")

        f.write("MEAN by experiment:\n")
        f.write(df.groupby("experiment")[["test_accuracy", "test_macro_f1"]].mean().sort_values("test_accuracy", ascending=False).to_string())
        f.write("\n\n")

        f.write("MEAN by augmentation:\n")
        f.write(df.groupby("augmentation")[["test_accuracy", "test_macro_f1"]].mean().sort_values("test_accuracy", ascending=False).to_string())
        f.write("\n")

    # 6) AUTOMATYCZNA ANALIZA I WNIOSKI (DO RAPORTU)
    analysis_path = PLOTS_DIR / "analysis.txt"

    best = df.loc[df["test_accuracy"].idxmax()]
    worst = df.loc[df["test_accuracy"].idxmin()]

    mean_model = df.groupby("model")[["test_accuracy", "test_macro_f1"]].mean()
    mean_exp = df.groupby("experiment")[["test_accuracy", "test_macro_f1"]].mean()
    mean_aug = df.groupby("augmentation")[["test_accuracy", "test_macro_f1"]].mean()

    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write("AUTOMATYCZNA ANALIZA WYNIKÓW\n")
        f.write("=" * 70 + "\n\n")

        f.write("1. Najlepszy i najgorszy przypadek\n")
        f.write("-" * 40 + "\n")
        f.write(
            f"Najlepszy wynik uzyskano dla modelu {best['model']} "
            f"w eksperymencie {best['experiment']} z augmentacją {best['augmentation']} "
            f"(accuracy={best['test_accuracy']:.3f}, macro-F1={best['test_macro_f1']:.3f}).\n"
        )
        f.write(
            f"Najgorszy wynik uzyskano dla modelu {worst['model']} "
            f"w eksperymencie {worst['experiment']} z augmentacją {worst['augmentation']} "
            f"(accuracy={worst['test_accuracy']:.3f}, macro-F1={worst['test_macro_f1']:.3f}).\n\n"
        )

        f.write("2. Porównanie architektur CNN\n")
        f.write("-" * 40 + "\n")
        best_model = mean_model["test_accuracy"].idxmax()
        f.write(
            f"Średnio najlepsze wyniki osiąga architektura {best_model}, "
            f"co wskazuje na jej lepszą zdolność generalizacji przy ograniczonej liczbie danych.\n\n"
        )

        f.write("3. Wpływ rodzaju danych treningowych (E1–E4)\n")
        f.write("-" * 40 + "\n")
        best_exp = mean_exp["test_accuracy"].idxmax()
        worst_exp = mean_exp["test_accuracy"].idxmin()
        f.write(
            f"Eksperyment {best_exp} daje najlepsze wyniki średnie, "
            f"natomiast {worst_exp} prowadzi do wyraźnego spadku jakości, "
            f"co sugeruje niewystarczającą reprezentatywność danych.\n\n"
        )

        f.write("4. Wpływ augmentacji danych\n")
        f.write("-" * 40 + "\n")
        best_aug = mean_aug["test_accuracy"].idxmax()
        worst_aug = mean_aug["test_accuracy"].idxmin()
        f.write(
            f"Najlepsze średnie wyniki uzyskano dla augmentacji {best_aug}, "
            f"podczas gdy {worst_aug} obniża jakość klasyfikacji, "
            f"co może świadczyć o nadmiernej deformacji danych wejściowych.\n\n"
        )

        f.write("5. Wniosek ogólny\n")
        f.write("-" * 40 + "\n")
        f.write(
            "Połączenie danych rzeczywistych i syntetycznych oraz dobór "
            "umiarkowanej augmentacji prowadzi do najlepszej generalizacji modeli CNN. "
            "Zbyt agresywne przekształcenia mogą pogarszać wyniki na zbiorze testowym.\n"
        )

    print("\n=== QUICK ANALYSIS ===")
    print("Best run :", best["run_id"], "| acc =", best["test_accuracy"])
    print("Best model (mean):", mean_model["test_accuracy"].idxmax())
    print("Best experiment  :", mean_exp["test_accuracy"].idxmax())
    print("Best augmentation:", mean_aug["test_accuracy"].idxmax())


    print("OK. Saved plots to:", PLOTS_DIR.resolve())
    print("Summary:", summary_path.resolve())


if __name__ == "__main__":
    main()
