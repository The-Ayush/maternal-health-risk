"""Step 2: Train and compare models, with and without duplicate leakage.

Run from the project root:  python src/02_train.py
Writes results/metrics.csv and figures/confusion_matrix.png.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

SEED = 42
DATA = Path("data/maternal_health_risk.csv")
FIG, RES = Path("figures"), Path("results")
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

LABELS = {"low risk": 0, "mid risk": 1, "high risk": 2}

raw = pd.read_csv(DATA, encoding="utf-8-sig")
raw["y"] = raw["RiskLevel"].map(LABELS)

# Cleaning: a heart rate of 7 bpm is a recording error.
raw = raw[raw["HeartRate"] >= 40]

clean = raw.drop_duplicates().reset_index(drop=True)
print(f"Rows after removing implausible values: {len(raw)}")
print(f"Unique rows after de-duplication:       {len(clean)}")

FEATURES = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]


def models():
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced")),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=SEED),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=SEED,
            eval_metric="mlogloss"),
    }


def evaluate(df, setting):
    X, y = df[FEATURES], df["y"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, preds = [], {}
    for name, model in models().items():
        p = cross_val_predict(model, X, y, cv=cv)
        preds[name] = p
        rows.append({
            "setting": setting,
            "model": name,
            "accuracy": (p == y).mean(),
            "macro_f1": f1_score(y, p, average="macro"),
            "high_risk_recall": recall_score(y, p, labels=[2], average="macro"),
        })
    return rows, preds


leaky_rows, _ = evaluate(raw, "with duplicates (leaky)")
clean_rows, clean_preds = evaluate(clean, "de-duplicated")

metrics = pd.DataFrame(leaky_rows + clean_rows).round(3)
metrics.to_csv(RES / "metrics.csv", index=False)
print("\n5-fold cross-validated results:\n")
print(metrics.to_string(index=False))

best = (metrics[metrics.setting == "de-duplicated"]
        .sort_values("macro_f1", ascending=False).iloc[0]["model"])
print(f"\nBest model on clean data: {best}")

ConfusionMatrixDisplay.from_predictions(
    clean["y"], clean_preds[best], display_labels=list(LABELS), cmap="Blues")
plt.title(f"{best} - de-duplicated data (5-fold CV)")
plt.tight_layout()
plt.savefig(FIG / "confusion_matrix.png", dpi=150)
plt.close()

# Per-setting summary for the README
gap = (metrics.pivot(index="model", columns="setting", values="macro_f1")
       .assign(inflation=lambda t: t["with duplicates (leaky)"] - t["de-duplicated"]))
print("\nMacro-F1 inflation caused by duplicate leakage:\n")
print(gap.round(3).to_string())
