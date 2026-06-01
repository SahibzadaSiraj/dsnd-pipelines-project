"""
StyleSense – Review Recommendation Prediction Pipeline
=======================================================
Predicts whether a customer would recommend a product (Recommended IND)
from review text, numerical, and categorical features.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    RocCurveDisplay, ConfusionMatrixDisplay, roc_curve
)
from sklearn.impute import SimpleImputer

# ── 1. LOAD DATA ────────────────────────────────────────────────────────────
print("=" * 60)
print("  StyleSense – Fashion Forward Forecasting Pipeline")
print("=" * 60)

df = pd.read_csv("/mnt/user-data/uploads/reviews.csv")
print(f"\n✓ Loaded {df.shape[0]:,} reviews, {df.shape[1]} columns")

# Drop rows where target is missing
df = df.dropna(subset=["Recommended IND"])
print(f"✓ {df.shape[0]:,} rows with valid target")

# ── 2. FEATURE ENGINEERING ──────────────────────────────────────────────────
# Combine title + review text for richer TF-IDF signal
df["full_text"] = (
    df["Title"].fillna("") + " " + df["Review Text"].fillna("")
).str.strip()

# Derived numeric: review length (proxy for engagement)
df["review_length"] = df["full_text"].str.split().str.len()

# ── 3. DEFINE FEATURE SETS ──────────────────────────────────────────────────
TEXT_COL    = "full_text"
NUM_COLS    = ["Age", "Positive Feedback Count", "review_length"]
CAT_COLS    = ["Division Name", "Department Name", "Class Name"]
TARGET      = "Recommended IND"

X = df[[TEXT_COL] + NUM_COLS + CAT_COLS]
y = df[TARGET]

print(f"\nClass distribution:")
vc = y.value_counts()
print(f"  Recommend (1): {vc[1]:,}  ({vc[1]/len(y):.1%})")
print(f"  Don't (0):     {vc[0]:,}  ({vc[0]/len(y):.1%})")

# ── 4. TRAIN / TEST SPLIT ───────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\n✓ Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── 5. PREPROCESSING ────────────────────────────────────────────────────────
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

text_transformer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    stop_words="english",
    min_df=3,
)

preprocessor = ColumnTransformer(transformers=[
    ("num",  numeric_transformer,    NUM_COLS),
    ("cat",  categorical_transformer, CAT_COLS),
    ("text", text_transformer,       TEXT_COL),
], remainder="drop")

# ── 6. MODELS ───────────────────────────────────────────────────────────────
models = {
    "Logistic Regression": Pipeline([
        ("prep", preprocessor),
        ("clf",  LogisticRegression(
            C=1.0, class_weight="balanced",
            max_iter=1000, solver="lbfgs", random_state=42
        )),
    ]),
    "Random Forest": Pipeline([
        ("prep", preprocessor),
        ("clf",  RandomForestClassifier(
            n_estimators=200, max_depth=15,
            class_weight="balanced", random_state=42, n_jobs=-1
        )),
    ]),
    "Gradient Boosting": Pipeline([
        ("prep", preprocessor),
        ("clf",  GradientBoostingClassifier(
            n_estimators=200, max_depth=4,
            learning_rate=0.1, random_state=42
        )),
    ]),
}

# ── 7. CROSS-VALIDATION COMPARISON ──────────────────────────────────────────
print("\n── Cross-Validation (5-fold, ROC-AUC) ──")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

for name, pipe in models.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                             scoring="roc_auc", n_jobs=-1)
    cv_results[name] = scores
    print(f"  {name:<25}  AUC = {scores.mean():.4f} ± {scores.std():.4f}")

# ── 8. TRAIN BEST MODEL & EVALUATE ON TEST SET ──────────────────────────────
best_name = max(cv_results, key=lambda k: cv_results[k].mean())
best_pipe  = models[best_name]

print(f"\n✓ Best model: {best_name}")
best_pipe.fit(X_train, y_train)

y_pred      = best_pipe.predict(X_test)
y_proba     = best_pipe.predict_proba(X_test)[:, 1]
test_auc    = roc_auc_score(y_test, y_proba)

print(f"\n── Test-Set Results ({best_name}) ──")
print(f"  ROC-AUC : {test_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=["Not Recommended", "Recommended"]))

# ── 9. VISUALISATIONS ───────────────────────────────────────────────────────
sns.set_style("whitegrid")
PALETTE = ["#E91E8C", "#7C3AED", "#06B6D4", "#10B981"]

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor("#0F0F1A")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

ax_cv   = fig.add_subplot(gs[0, 0])
ax_roc  = fig.add_subplot(gs[0, 1])
ax_cm   = fig.add_subplot(gs[0, 2])
ax_feat = fig.add_subplot(gs[1, :2])
ax_dist = fig.add_subplot(gs[1, 2])

DARK   = "#0F0F1A"
PANEL  = "#1A1A2E"
LIGHT  = "#E2E8F0"
ACCENT = "#E91E8C"

def style_ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=LIGHT, labelsize=9)
    ax.xaxis.label.set_color(LIGHT)
    ax.yaxis.label.set_color(LIGHT)
    ax.title.set_color(LIGHT)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2D2D44")

# 9a – CV AUC comparison
means = [cv_results[n].mean() for n in models]
stds  = [cv_results[n].std()  for n in models]
bars  = ax_cv.barh(list(models.keys()), means, xerr=stds,
                   color=PALETTE[:3], edgecolor="none",
                   error_kw=dict(ecolor=LIGHT, capsize=4))
ax_cv.set_xlim(0.7, 1.0)
ax_cv.set_xlabel("ROC-AUC")
ax_cv.set_title("5-Fold CV – Model Comparison", fontweight="bold")
for bar, m in zip(bars, means):
    ax_cv.text(m + 0.002, bar.get_y() + bar.get_height()/2,
               f"{m:.4f}", va="center", color=LIGHT, fontsize=9)
style_ax(ax_cv)

# 9b – ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
ax_roc.plot(fpr, tpr, color=ACCENT, lw=2,
            label=f"{best_name}\nAUC = {test_auc:.4f}")
ax_roc.plot([0,1],[0,1], "--", color="#555577", lw=1)
ax_roc.fill_between(fpr, tpr, alpha=0.15, color=ACCENT)
ax_roc.set_xlabel("False Positive Rate"); ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("ROC Curve – Test Set", fontweight="bold")
ax_roc.legend(facecolor=PANEL, edgecolor="#2D2D44",
               labelcolor=LIGHT, fontsize=9)
style_ax(ax_roc)

# 9c – Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="RdPu",
            xticklabels=["Not Rec.", "Rec."],
            yticklabels=["Not Rec.", "Rec."],
            linewidths=0.5, linecolor="#0F0F1A",
            cbar=False, ax=ax_cm)
ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("Actual")
ax_cm.set_title("Confusion Matrix", fontweight="bold")
style_ax(ax_cm)

# 9d – Top TF-IDF features (only for Logistic Regression / if available)
if "Logistic Regression" in models:
    lr_pipe = models["Logistic Regression"]
    lr_pipe.fit(X_train, y_train)
    ct   = lr_pipe.named_steps["prep"]
    clf  = lr_pipe.named_steps["clf"]
    
    text_feat = ct.named_transformers_["text"].get_feature_names_out()
    num_feat  = NUM_COLS
    cat_feat  = ct.named_transformers_["cat"].named_steps["ohe"].get_feature_names_out(CAT_COLS)
    all_feat  = list(num_feat) + list(cat_feat) + list(text_feat)
    
    coef = clf.coef_[0]
    n_show = 20
    top_pos_idx = np.argsort(coef)[-n_show:][::-1]
    top_neg_idx = np.argsort(coef)[:n_show]
    
    idxs   = np.concatenate([top_pos_idx, top_neg_idx])
    values = coef[idxs]
    names  = [all_feat[i] for i in idxs]
    colors = [PALETTE[0] if v > 0 else PALETTE[2] for v in values]
    
    ax_feat.barh(range(len(names)), values, color=colors, edgecolor="none")
    ax_feat.set_yticks(range(len(names)))
    ax_feat.set_yticklabels(names, fontsize=8)
    ax_feat.axvline(0, color=LIGHT, lw=0.8, linestyle="--")
    ax_feat.set_title("Top Feature Coefficients (Logistic Regression)",
                       fontweight="bold")
    ax_feat.set_xlabel("Coefficient (→ Recommend)")
    
    from matplotlib.patches import Patch
    legend_els = [Patch(color=PALETTE[0], label="Signals Recommendation"),
                  Patch(color=PALETTE[2], label="Signals Non-Recommendation")]
    ax_feat.legend(handles=legend_els, facecolor=PANEL,
                   edgecolor="#2D2D44", labelcolor=LIGHT, fontsize=8)
    style_ax(ax_feat)

# 9e – Predicted probability distribution
ax_dist.hist(y_proba[y_test==1], bins=40, color=PALETTE[0],
             alpha=0.7, label="Recommended (1)", density=True)
ax_dist.hist(y_proba[y_test==0], bins=40, color=PALETTE[2],
             alpha=0.7, label="Not Recommended (0)", density=True)
ax_dist.axvline(0.5, color=LIGHT, lw=1.2, linestyle="--")
ax_dist.set_xlabel("Predicted Probability")
ax_dist.set_ylabel("Density")
ax_dist.set_title("Predicted Probability Distribution", fontweight="bold")
ax_dist.legend(facecolor=PANEL, edgecolor="#2D2D44",
               labelcolor=LIGHT, fontsize=8)
style_ax(ax_dist)

# Title
fig.suptitle("StyleSense – Recommendation Prediction Results",
             fontsize=16, fontweight="bold", color=LIGHT, y=0.98)

plt.savefig("/mnt/user-data/outputs/stylesense_results.png",
            dpi=150, bbox_inches="tight", facecolor=DARK)
print("\n✓ Dashboard saved → stylesense_results.png")

# ── 10. SUMMARY ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PIPELINE SUMMARY")
print("=" * 60)
print(f"  Models evaluated    : {len(models)}")
print(f"  Best model          : {best_name}")
print(f"  Test ROC-AUC        : {test_auc:.4f}")
from sklearn.metrics import accuracy_score, f1_score
print(f"  Test Accuracy       : {accuracy_score(y_test, y_pred):.4f}")
print(f"  Test F1 (weighted)  : {f1_score(y_test, y_pred, average='weighted'):.4f}")
print("=" * 60)
