"""Step 1: Exploratory data analysis of the Maternal Health Risk dataset.

Run from the project root:  python src/01_eda.py
Plots are saved to figures/.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATA = Path("data/maternal_health_risk.csv")
FIG = Path("figures")
FIG.mkdir(exist_ok=True)

df = pd.read_csv(DATA, encoding="utf-8-sig")

print("Shape:", df.shape)
print("\nMissing values per column:\n", df.isna().sum())
print("\nSummary statistics:\n", df.describe().round(2))
print("\nClass balance:\n", df["RiskLevel"].value_counts())
print("\nExact duplicate rows:", df.duplicated().sum())

# Class balance
sns.countplot(data=df, x="RiskLevel", order=["low risk", "mid risk", "high risk"])
plt.title("Risk level distribution")
plt.tight_layout()
plt.savefig(FIG / "class_balance.png", dpi=150)
plt.close()

# Feature distributions by risk level
features = df.columns.drop("RiskLevel")
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, col in zip(axes.flat, features):
    sns.boxplot(data=df, x="RiskLevel", y=col, ax=ax,
                order=["low risk", "mid risk", "high risk"])
    ax.set_title(col)
plt.tight_layout()
plt.savefig(FIG / "features_by_risk.png", dpi=150)
plt.close()

# Correlations between features
sns.heatmap(df[features].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature correlations")
plt.tight_layout()
plt.savefig(FIG / "correlations.png", dpi=150)
plt.close()

print("\nSaved plots to", FIG.resolve())
