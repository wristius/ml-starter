"""Run with:  python -m mlstarter"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .evaluate import run


def main() -> None:
    results, best_name, report = run()
    out = Path("outputs")
    out.mkdir(exist_ok=True)

    results.to_csv(out / "model_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(results["model"], results["cv_mean"], xerr=results["cv_std"])
    ax.set_xlim(0.8, 1.0)
    ax.set_xlabel("5-fold CV accuracy")
    ax.invert_yaxis()
    ax.set_title("Baseline model comparison (Wine dataset)")
    fig.tight_layout()
    fig.savefig(out / "model_comparison.png", dpi=150)

    print(results.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nBest by CV: {best_name}\n")
    print("Held-out test report:\n" + report)


if __name__ == "__main__":
    main()
