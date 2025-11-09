
# AMME5710 - Computer Vision and Image Processing - Major Project
# Authors: Varunvarshan Sideshkumar, Arin Adurkar, Siwon Kang

# Purpose of this code:
#   1) Load gestures CSV -> X=numeric, y='label'.
#   2) 80/20 stratified split, 5-fold CV on train.
#   3) Train SVM/KNN/DT/RF, pick winner by average of (CV F1_macro, Test F1_macro).
#   4) Save best model, scaler, label map, feature order, metrics table.
#   5) Produce richer plots:
#        - confusion + normalized confusion (per model)
#        - PR curves + PR-AUC (per model)
#        - ROC curves OvR + micro/macro AUC (per model)
#        - reliability (calibration) + ECE (per model)
#        - feature correlation heatmap (dataset-level)
#        - PCA 2-D scatter on test split (dataset-level)
#        - learning curve for best model
#        - feature importance (model-based or permutation) for best model


import numpy as np
import pandas as pd
import pickle
from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    learning_curve,
)
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
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    auc,
)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns  # only for corr heatmap; remove if not desired

# IO paths
CSV_PATH   = Path("data/gestures_log.csv")
SCALER_PKL = Path("gesture_scaler.pkl")
MODEL_PKL  = Path("gesture_model.pkl")
LABEL_MAP  = Path("label_map.npy")
FEAT_NAMES = Path("feature_names.npy")
OUT_CSV    = Path("metrics_summary.csv")
REPORT_DIR = Path("Training_Model_Results")
CM_DIR     = REPORT_DIR / "confusion_matrices"
PR_DIR     = REPORT_DIR / "pr_curves"
ROC_DIR    = REPORT_DIR / "roc_curves"
CAL_DIR    = REPORT_DIR / "calibration"
BEST_DIR   = REPORT_DIR / "best_model"
DATA_DIR   = REPORT_DIR / "dataset_plots"

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
    return X, y, le, feat_names, X_num  # return X_num (DataFrame) for dataset-level plots

def cv_metric(model, X, y, cv, scorer):
    return float(np.mean(cross_val_score(model, X, y, cv=cv, scoring=scorer, n_jobs=-1)))

def save_confusion_matrix(y_true, y_pred, class_names, title, out_path, normalize=False):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    if normalize:
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None),
            display_labels=class_names,
        )
    else:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=140)
    disp.plot(ax=ax, values_format=('d' if not normalize else '.2f'), cmap="Blues", colorbar=True)
    ax.set_title(title)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

def save_pr_curves(y_true, prob_mat, class_names, model_name):
    PR_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 7))
    ap_scores = []
    for c in range(len(class_names)):
        y_bin = (y_true == c).astype(np.uint8)
        precision, recall, _ = precision_recall_curve(y_bin, prob_mat[:, c])
        ap = average_precision_score(y_bin, prob_mat[:, c])
        ap_scores.append(ap)
        plt.plot(recall, precision, label=f"{class_names[c]} (AP={ap:.2f})")
    plt.xlabel("recall"); plt.ylabel("precision"); plt.title(f"{model_name} PR curves")
    plt.legend(loc="lower left", ncol=2, fontsize=8)
    fig.savefig(PR_DIR / f"{model_name}_pr_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8, 4))
    xs = np.arange(len(class_names))
    plt.bar(xs, ap_scores)
    plt.xticks(xs, class_names, rotation=45, ha="right")
    plt.ylabel("AP (PR-AUC)")
    plt.title(f"{model_name} per-class PR-AUC")
    fig.tight_layout()
    fig.savefig(PR_DIR / f"{model_name}_pr_auc_bars.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

def save_roc_curves(y_true, prob_mat, class_names, model_name):
    ROC_DIR.mkdir(parents=True, exist_ok=True)
    C = len(class_names)
    fig = plt.figure(figsize=(8, 7))

    y_true_bin = np.zeros_like(prob_mat)
    for i, c in enumerate(y_true):
        y_true_bin[i, c] = 1
    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), prob_mat.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    plt.plot(fpr_micro, tpr_micro, label=f"micro-avg (AUC={auc_micro:.2f})", linewidth=2)

    aucs = []
    for c in range(C):
        y_bin = (y_true == c).astype(np.uint8)
        fpr, tpr, _ = roc_curve(y_bin, prob_mat[:, c])
        auc_c = auc(fpr, tpr)
        aucs.append(auc_c)
        plt.plot(fpr, tpr, label=f"{class_names[c]} (AUC={auc_c:.2f})", alpha=0.8)

    auc_macro = float(np.mean(aucs)) if aucs else 0.0
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"{model_name} ROC (macro AUC={auc_macro:.2f})")
    plt.legend(loc="lower right", fontsize=8, ncol=2)
    fig.savefig(ROC_DIR / f"{model_name}_roc_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

def save_calibration_plot(y_true, prob_mat, model_name):
    CAL_DIR.mkdir(parents=True, exist_ok=True)
    conf = prob_mat.max(axis=1)
    correct = (y_true == np.argmax(prob_mat, axis=1)).astype(np.uint8)
    prob_true, prob_pred = calibration_curve(correct, conf, n_bins=10, strategy="uniform")
    ece = float(np.mean(np.abs(prob_pred - prob_true)))
    fig = plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="perfect")
    plt.plot(prob_pred, prob_true, marker="o", label=f"model (ECE≈{ece:.03f})")
    plt.xlabel("predicted confidence"); plt.ylabel("observed accuracy")
    plt.title(f"{model_name} reliability")
    plt.legend()
    fig.savefig(CAL_DIR / f"{model_name}_calibration.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

def evaluate_models(Xs, y, le):
    X_train, X_test, y_train, y_test = train_test_split(
        Xs, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "SVM": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True),
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
    PR_DIR.mkdir(parents=True, exist_ok=True)
    ROC_DIR.mkdir(parents=True, exist_ok=True)
    CAL_DIR.mkdir(parents=True, exist_ok=True)

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

        save_confusion_matrix(
            y_test, y_pred, list(le.classes_),
            f"{name} Confusion Matrix",
            CM_DIR / f"{name}_cm.png",
            normalize=False
        )
        save_confusion_matrix(
            y_test, y_pred, list(le.classes_),
            f"{name} Confusion Matrix (row-normalized)",
            CM_DIR / f"{name}_cm_norm.png",
            normalize=True
        )

        # probabilistic curves 
        prob_ok = hasattr(model, "predict_proba")
        if prob_ok:
            prob_mat = model.predict_proba(X_test)
        else:

            if hasattr(model, "decision_function"):
                df = model.decision_function(X_test)
                # min-max normalize per row
                mn = df.min(axis=1, keepdims=True)
                mx = df.max(axis=1, keepdims=True)
                prob_mat = (df - mn) / np.clip(mx - mn, 1e-9, None)
            else:
                prob_mat = None

        if prob_mat is not None and prob_mat.ndim == 2 and prob_mat.shape[1] == len(le.classes_):
            save_pr_curves(y_test, prob_mat, list(le.classes_), name)
            save_roc_curves(y_test, prob_mat, list(le.classes_), name)
            save_calibration_plot(y_test, prob_mat, name)

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
    return metrics_df, best_name, best_model, X_train, X_test, y_train, y_test

def plot_learning_curve(estimator, X, y, title, out_path):
    fig = plt.figure(figsize=(7,5))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, scoring="f1_macro", n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 6), shuffle=True, random_state=42
    )
    tr_mean = train_scores.mean(axis=1); tr_std = train_scores.std(axis=1)
    va_mean = val_scores.mean(axis=1);   va_std = val_scores.std(axis=1)
    plt.fill_between(train_sizes, tr_mean-tr_std, tr_mean+tr_std, alpha=0.2)
    plt.fill_between(train_sizes, va_mean-va_std, va_mean+va_std, alpha=0.2)
    plt.plot(train_sizes, tr_mean, marker="o", label="train F1_macro")
    plt.plot(train_sizes, va_mean, marker="s", label="CV F1_macro")
    plt.xlabel("training examples"); plt.ylabel("F1_macro"); plt.title(title)
    plt.legend(); fig.tight_layout(); fig.savefig(out_path, dpi=160); plt.close(fig)

def plot_feature_importance(best_model, feat_names, X_test, y_test, out_path_model, out_path_perm):
    # model-based (DT/RF)
    if hasattr(best_model, "feature_importances_"):
        imp = np.array(best_model.feature_importances_, dtype=float)
        order = np.argsort(imp)[::-1][:30]
        fig = plt.figure(figsize=(8, max(4, 0.25*len(order)+1)))
        plt.barh(np.array(feat_names)[order][::-1], imp[order][::-1])
        plt.title("Model-based feature importance")
        plt.xlabel("importance")
        plt.tight_layout(); fig.savefig(out_path_model, dpi=160, bbox_inches="tight"); plt.close(fig)
    # permutation (model-agnostic)
    try:
        r = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1, scoring="f1_macro")
        imp = r.importances_mean
        order = np.argsort(imp)[::-1][:30]
        fig = plt.figure(figsize=(8, max(4, 0.25*len(order)+1)))
        plt.barh(np.array(feat_names)[order][::-1], imp[order][::-1])
        plt.title("Permutation feature importance (F1_macro)")
        plt.xlabel("importance")
        plt.tight_layout(); fig.savefig(out_path_perm, dpi=160, bbox_inches="tight"); plt.close(fig)
    except Exception as e:
        print(f"[warn] permutation importance failed: {e}")

def dataset_level_plots(X_num_df: pd.DataFrame, y_enc: np.ndarray, class_names: list):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # correlation heatmap
    try:
        corr = X_num_df.corr(numeric_only=True)
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(corr, cmap="coolwarm", center=0, square=False, cbar_kws={"shrink":0.8})
        plt.title("Feature correlation heatmap")
        fig.savefig(DATA_DIR / "correlation_heatmap.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    except Exception as e:
        print(f"[warn] correlation heatmap skipped: {e}")

def pca_scatter(X_test, y_test, class_names):
    try:
        pca = PCA(n_components=2, random_state=42)
        Z = pca.fit_transform(X_test)
        fig = plt.figure(figsize=(7,6))
        for c in range(len(class_names)):
            m = (y_test == c)
            if np.any(m):
                plt.scatter(Z[m,0], Z[m,1], s=16, alpha=0.7, label=class_names[c])
        plt.legend(fontsize=8, ncol=2)
        plt.title(f"PCA scatter (explained={pca.explained_variance_ratio_.sum():.2f})")
        plt.xlabel("PC1"); plt.ylabel("PC2")
        fig.savefig(DATA_DIR / "pca_scatter_test.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    except Exception as e:
        print(f"[warn] PCA scatter skipped: {e}")

def train_and_evaluate():
    X, y, le, feat_names, X_num_df = load_data(CSV_PATH)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # dataset-level plots (before split)
    dataset_level_plots(X_num_df, y, list(le.classes_))

    metrics_df, best_name, best_model, X_train, X_test, y_train, y_test = evaluate_models(Xs, y, le)
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

    # learning curve + feature importance for the final chosen model
    BEST_DIR.mkdir(parents=True, exist_ok=True)
    plot_learning_curve(best_model, X_train, y_train, f"{best_name} learning curve (F1_macro)", BEST_DIR / "learning_curve.png")
    plot_feature_importance(
        best_model, feat_names, X_test, y_test,
        BEST_DIR / "feature_importance_model.png",
        BEST_DIR / "feature_importance_perm.png"
    )

    # PCA scatter on test split
    pca_scatter(X_test, y_test, list(le.classes_))

if __name__ == "__main__":
    train_and_evaluate()
