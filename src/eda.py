"""Who churns? A few plots before modelling.

Run:  python -m src.eda
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.data import FIGURES_DIR, TARGET, load_data


def churn_rate_plot(df, col, ax):
    rates = df.groupby(col)[TARGET].mean().sort_values(ascending=False) * 100
    rates.plot.bar(ax=ax, color="#C44E52")
    ax.set_ylabel("Churn rate (%)")
    ax.set_xlabel("")
    ax.set_title(col)
    ax.tick_params(axis="x", rotation=20)
    for i, v in enumerate(rates):
        ax.text(i, v + 0.8, f"{v:.0f}%", ha="center", fontsize=9)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    df = load_data()
    print("Shape:", df.shape)
    print(f"Overall churn rate: {df[TARGET].mean():.1%}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, col in zip(axes.flat, ["Contract", "InternetService", "PaymentMethod", "TenureGroup"]):
        churn_rate_plot(df, col, ax)
    fig.suptitle("Churn rate by customer segment", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "churn_by_segment.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col in zip(axes, ["tenure", "MonthlyCharges"]):
        sns.kdeplot(data=df, x=col, hue=TARGET, common_norm=False, fill=True, ax=ax, palette=["#4C72B0", "#C44E52"])
        ax.set_title(f"{col}: churners vs stayers")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "numeric_distributions.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    df[TARGET].map({0: "Stayed", 1: "Churned"}).value_counts().plot.pie(
        ax=ax, autopct="%1.1f%%", colors=["#4C72B0", "#C44E52"], startangle=90)
    ax.set_ylabel("")
    ax.set_title("Class balance")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_balance.png", dpi=120)
    plt.close(fig)

    for col in ["Contract", "InternetService", "PaymentMethod", "TenureGroup"]:
        print(f"\n{col}:\n", (df.groupby(col)[TARGET].mean() * 100).round(1).sort_values(ascending=False))
    print(f"\nFigures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
