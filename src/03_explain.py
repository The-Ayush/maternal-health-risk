"""Step 3: Explain the XGBoost model with SHAP.

Run from the project root:  python src/03_explain.py
Fits XGBoost on a train split of the de-duplicated data, then saves:
  figures/shap_summary_high_risk.png  - global feature importance for 'high risk'
  figures/shap_patient_example.png    - why one test patient was flagged high risk
  results/feature_importance.csv      - mean |SHAP| per feature and class
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

SEED = 42
FIG, RES = Path("figures"), Path("results")
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

LABELS = {"low risk": 0, "mid risk": 1, "high risk": 2}
FEATURES = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]

df = pd.read_csv("data/maternal_health_risk.csv", encoding="utf-8-sig")
df = df[df["HeartRate"] >= 40].drop_duplicates().reset_index(drop=True)
df["y"] = df["RiskLevel"].map(LABELS)

X_tr, X_te, y_tr, y_te = train_test_split(
    df[FEATURES], df["y"], test_size=0.25, stratify=df["y"], random_state=SEED)

model = XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9,
    colsample_bytree=0.9, random_state=SEED, eval_metric="mlogloss")
model.fit(X_tr, y_tr)
print(classification_report(y_te, model.predict(X_te), target_names=list(LABELS)))

explainer = shap.TreeExplainer(model)
sv = explainer(X_te)  # shape: (n_samples, n_features, n_classes)

# Global importance per class
imp = pd.DataFrame(np.abs(sv.values).mean(axis=0), index=FEATURES, columns=list(LABELS))
imp = imp.sort_values("high risk", ascending=False).round(3)
imp.to_csv(RES / "feature_importance.csv")
print("Mean |SHAP| per feature and class:\n", imp)

# Beeswarm for the high-risk class
shap.plots.beeswarm(sv[:, :, 2], show=False)
plt.title("SHAP values for the 'high risk' class")
plt.tight_layout()
plt.savefig(FIG / "shap_summary_high_risk.png", dpi=150)
plt.close()

# Explain one correctly flagged high-risk patient
proba = model.predict_proba(X_te)[:, 2]
hits = np.where((y_te.values == 2) & (proba > 0.5))[0]
i = hits[np.argmax(proba[hits])]
shap.plots.waterfall(sv[i, :, 2], show=False)
plt.title(f"Why this patient was flagged high risk (p = {proba[i]:.2f})")
plt.tight_layout()
plt.savefig(FIG / "shap_patient_example.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved SHAP plots to", FIG.resolve())
