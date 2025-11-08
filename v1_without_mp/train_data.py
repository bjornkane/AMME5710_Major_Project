# train_models.py
#
# Trains SVM/KNN/DT/RF on gestures_all.csv.
# X = ONLY numeric columns, y = 'label'.
# Saves:
#   gesture_scaler.pkl
#   gesture_model.pkl
#   label_map.npy
#   feature_names.npy
#   metrics_summary.csv
#   reports/<Model>_report.txt
#   reports/confusion_matrices/<Model>_cm.png  <-- added
#
# Usage:
#   python train_models.py

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import matplotlib
matplotlib.use("Agg")  # headless save
import matplotlib.pyplot as plt

CSV_PATH   = Path("gestures_all.csv")
SCALER_PKL = Path("gesture_scaler.pkl")
MODEL_PKL  = Path("gesture_model.pkl")
LABEL_MAP  = Path("label_map.npy")
FEAT_NAMES = Path("feature_names.npy")
OUT_CSV    = Path("metrics_summary.csv")
REPORT_DIR = Path("reports")
CM_DIR     = REPORT_DIR / "confusion_matrices"

def load_data(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")

    y_raw = df["label"].astype(str).values
    X_df = df.drop(columns=["label"])
    X_num = X_df.select_dtypes(include=[np.number]).copy()

    mask_valid = ~X_num.isna().any(axis=1)
    if (~mask_valid).any():
        dropped = int((~mask_valid).sum())
        print(f"[clean] dropped rows with NaNs in numeric features: {dropped}")

    X_num = X_num[mask_valid]
    y_raw = y_raw[mask_valid.values]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    X = X_num.to_numpy(dtype=np.float32)
    feat_names = list(X_num.columns)

    print("\n=== Classes Loaded ===")
    for i, c in enumerate(le.classes_):
        print(f"{i}: {c}")
    print(f"\n[load] samples={X.shape[0]} features={X.shape[1]}")
    print(f"[load] feature_cols={feat_names}\n")
    return X, y, le, feat_names

def cv_metric(model, X, y, cv, scorer):
    return float(np.mean(cross_val_score(model, X, y, cv=cv, scoring=scorer, n_jobs=-1)))

def save_confusion_matrix(y_true, y_pred, class_names, title, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=140)
    disp.plot(ax=ax, values_format='d', cmap="Blues", colorbar=True)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

def evaluate_models(Xs, y, le):
    X_train, X_test, y_train, y_test = train_test_split(
        Xs, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "SVM": SVC(kernel="rbf", C=1.0, gamma="scale"),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "DT" : DecisionTreeClassifier(max_depth=8, random_state=42),
        "RF" : RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    best_score = -1.0
    best_model = None
    best_name = ""

    REPORT_DIR.mkdir(exist_ok=True)
    CM_DIR.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        cv_f1 = cv_metric(model, X_train, y_train, cv, "f1_macro")
        cv_acc = cv_metric(model, X_train, y_train, cv, "accuracy")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        test_acc = accuracy_score(y_test, y_pred)
        _, _, f1_m, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )

        report_txt = classification_report(
            y_test, y_pred, target_names=list(le.classes_), digits=4, zero_division=0
        )
        print("\n======================================================")
        print(f"Model: {name}  (Test Classification Report)")
        print("======================================================")
        print(report_txt)
        (REPORT_DIR / f"{name}_report.txt").write_text(report_txt, encoding="utf-8")

        # save confusion matrix image
        cm_path = CM_DIR / f"{name}_cm.png"
        save_confusion_matrix(y_test, y_pred, list(le.classes_), f"{name} Confusion Matrix", cm_path)
        print(f"[save] confusion matrix -> {cm_path}")

        rows.append({
            "Model": name,
            "CV_Acc": cv_acc,
            "CV_F1_macro": cv_f1,
            "Test_Acc": test_acc,
            "Test_F1_macro": f1_m,
        })

        score = 0.5 * (cv_f1 + f1_m)
        if score > best_score:
            best_score = score
            best_model = model
            best_name = name

    metrics_df = pd.DataFrame(rows).sort_values(by="Test_F1_macro", ascending=False)
    return metrics_df, best_name, best_model

def train_and_evaluate():
    X, y, le, feat_names = load_data(CSV_PATH)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    metrics_df, best_name, best_model = evaluate_models(Xs, y, le)
    metrics_df.to_csv(OUT_CSV, index=False)

    print("\n================= Comparison Table =================")
    print(metrics_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n=== Final Class Order Used for Model Encoding ===")
    for i, c in enumerate(le.classes_):
        print(f"{i}: {c}")

    with open(SCALER_PKL, "wb") as f:
        pickle.dump(scaler, f)
    with open(MODEL_PKL, "wb") as f:
        pickle.dump(best_model, f)
    np.save(LABEL_MAP, le.classes_)
    np.save(FEAT_NAMES, np.array(feat_names, dtype=object))

    print(f"\n[save] scaler  -> {SCALER_PKL}")
    print(f"[save] model   -> {MODEL_PKL}")
    print(f"[save] labels  -> {LABEL_MAP}")
    print(f"[save] feature -> {FEAT_NAMES}")
    print(f"[save] metrics -> {OUT_CSV}")
    print(f"Best Model: {best_name}")

if __name__ == "__main__":
    train_and_evaluate()
