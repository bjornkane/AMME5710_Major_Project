# benchmark_from_dataset_resnet18.py
#
# Student roadmap (what/why):
#   1) Pack "dataset/<class>/<view>/*" into ImageFolder at "benchmarking/dataset_flat/<class>/*".
#   2) Split into train/val/test; fine-tune ResNet-18 (ImageNet init).
#   3) Save best weights + ONNX + text/JSON reports.
#   4) Extra benchmarks: norm CM, PR/ROC, calibration (ECE), top-K, class support,
#      misclassification gallery, t-SNE of penultimate features.
#   5) No CLI flags — everything in CONFIG.
#
# How to run:
#   python benchmark_from_dataset_resnet18.py
#
# Expected input layout:
#   dataset/
#     open_palm/    {images_crop|images_raw|masks|overlays}/*.jpg
#     closed_fist/  {images_crop|images_raw|masks|overlays}/*.jpg
#
# Outputs (under "benchmarking/resnet18_<VIEW>/"): weights, ONNX, plots, reports.

import os, json, random, shutil
from pathlib import Path
from math import ceil

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torchvision.utils import make_grid
from PIL import Image

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    auc,
)
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE

# ============================ CONFIG (edit here) ============================
CONFIG = dict(
    SRC_ROOT="data",          # where class folders live (relative to this .py)
    VIEW="images_crop",          # "images_crop" | "images_raw" | "masks" | "overlays"
    IMG_SIZE=224,                # ResNet-18 default side
    EPOCHS=20,                   # bump if needed
    BATCH_SIZE=32,               # reduce if VRAM is tight
    LR=1e-3,                     # AdamW base LR
    WD=1e-4,                     # weight decay
    SEED=42,                     # reproducibility
    PACK_MODE="copy",            # "copy" | "move" | "link"
    VAL_FRAC=0.15,               # 15% validation
    TEST_FRAC=0.15,              # 15% test
    NUM_WORKERS=0,               # Windows-friendly default; raise if your env supports it
    MISCLS_MAX=24,               # max images in misclassification gallery
)
# ===========================================================================


# ---------- basic utilities ----------
def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def is_image(p: Path):
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def copy_or_link(src: Path, dst: Path, mode: str):
    if mode == "move":
        shutil.move(str(src), str(dst))
    elif mode == "link":
        try:
            os.link(src, dst)  # hard link if same volume
        except Exception:
            shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)


# ---------- dataset packer (ImageFolder builder) ----------
def build_imagefolder_from_dataset(src_root: Path, dst_root: Path, view: str, mode="copy"):
    """
    For each class folder, choose the requested 'view'. If missing, fall back:
      prefer images_crop → fallback images_raw (or vice versa if VIEW == images_crop).
    """
    safe_mkdir(dst_root)
    class_counts = {}
    classes = []

    # guard: no iterdir() if source doesn't exist
    if not src_root.exists():
        return [], {}

    for cls_dir in sorted([p for p in src_root.iterdir() if p.is_dir()]):
        cls = cls_dir.name
        src_view = cls_dir / view

        if not src_view.exists():
            fallback = (cls_dir / "images_crop" if view != "images_crop" else cls_dir / "images_raw")
            if fallback.exists():
                src_view = fallback
            else:
                print(f"[skip] {cls}: no '{view}' or fallback view found.")
                continue

        classes.append(cls)
        out_dir = dst_root / cls
        safe_mkdir(out_dir)

        n = 0
        for imgp in src_view.rglob("*"):
            if imgp.is_file() and is_image(imgp):
                dst = out_dir / f"{cls}_{imgp.stem}{imgp.suffix.lower()}"
                if not dst.exists():
                    copy_or_link(imgp, dst, mode)
                n += 1

        class_counts[cls] = n
        print(f"[pack] {cls:<16} view={src_view.name:<12} -> {n:5d} files")

    classes = sorted(classes)
    return classes, class_counts


# ---------- plotting helpers ----------
def plot_confusion(cm, class_names, out_png, normalize=False):
    cm_plot = cm.astype(np.float32)
    if normalize and cm.sum(axis=1).min() > 0:
        cm_plot = cm_plot / cm_plot.sum(axis=1, keepdims=True)

    fig = plt.figure(figsize=(8, 7))
    im = plt.imshow(cm_plot, interpolation="nearest", cmap="Blues")
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("norm" if normalize else "count")
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)

    if not normalize:
        thresh = cm_plot.max() / 2.0 if cm_plot.size > 0 else 0.5
        for i in range(cm_plot.shape[0]):
            for j in range(cm_plot.shape[1]):
                txt = format(int(cm[i, j]), "d")
                plt.text(
                    j, i, txt, ha="center", va="center",
                    color="white" if cm_plot[i, j] > thresh else "black",
                )

    plt.tight_layout()
    plt.ylabel("true")
    plt.xlabel("pred")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curves(y_true, y_prob, class_names, out_png, out_auc_bars):
    fig = plt.figure(figsize=(8, 7))
    C = y_prob.shape[1]
    ap_scores = []
    for c in range(C):
        y_bin = (y_true == c).astype(np.uint8)
        precision, recall, _ = precision_recall_curve(y_bin, y_prob[:, c])
        ap = average_precision_score(y_bin, y_prob[:, c])
        ap_scores.append(ap)
        plt.plot(recall, precision, label=f"{class_names[c]} (AP={ap:.2f})")
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.title("Per-class PR curves")
    plt.legend(loc="lower left", ncol=2, fontsize=8)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8, 4))
    xs = np.arange(len(class_names))
    plt.bar(xs, ap_scores)
    plt.xticks(xs, class_names, rotation=45, ha="right")
    plt.ylabel("AP (PR-AUC)")
    plt.title("Per-class PR-AUC")
    fig.tight_layout()
    fig.savefig(out_auc_bars, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(y_true, y_prob, class_names, out_png):
    C = y_prob.shape[1]
    fig = plt.figure(figsize=(8, 7))

    y_true_bin = np.zeros_like(y_prob)
    for i, c in enumerate(y_true):
        y_true_bin[i, c] = 1
    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    plt.plot(fpr_micro, tpr_micro, label=f"micro-avg ROC (AUC={auc_micro:.2f})", linewidth=2)

    aucs = []
    for c in range(C):
        y_bin = (y_true == c).astype(np.uint8)
        fpr, tpr, _ = roc_curve(y_bin, y_prob[:, c])
        auc_c = auc(fpr, tpr)
        aucs.append(auc_c)
        plt.plot(fpr, tpr, label=f"{class_names[c]} (AUC={auc_c:.2f})", alpha=0.8)

    auc_macro = float(np.mean(aucs)) if aucs else 0.0
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC (macro AUC={auc_macro:.2f})")
    plt.legend(loc="lower right", fontsize=8, ncol=2)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(y_true, y_prob, out_png, n_bins=10):
    conf = y_prob.max(axis=1)
    correct = (y_true == np.argmax(y_prob, axis=1)).astype(np.uint8)
    prob_true, prob_pred = calibration_curve(correct, conf, n_bins=n_bins, strategy="uniform")
    ece = np.mean(np.abs(prob_pred - prob_true))
    fig = plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="perfect")
    plt.plot(prob_pred, prob_true, marker="o", label=f"model (ECE≈{ece:.03f})")
    plt.xlabel("predicted confidence")
    plt.ylabel("observed accuracy")
    plt.title("Reliability (Calibration) diagram")
    plt.legend()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_topk_curve(logits, y_true, out_png, k_max=5):
    ks = list(range(1, k_max + 1))
    accs = []
    for k in ks:
        topk = torch.topk(torch.tensor(logits), k=k, dim=1).indices.numpy()
        hits = np.any(topk == y_true[:, None], axis=1).mean() if len(y_true) else 0.0
        accs.append(hits)
    fig = plt.figure(figsize=(6, 4))
    plt.plot(ks, accs, marker="o")
    plt.xticks(ks)
    plt.ylim(0, 1)
    plt.xlabel("k")
    plt.ylabel("top-k accuracy")
    plt.title("Top-K accuracy curve")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_class_support(y_true, class_names, out_png):
    counts = np.bincount(y_true, minlength=len(class_names))
    fig = plt.figure(figsize=(8, 4))
    xs = np.arange(len(class_names))
    plt.bar(xs, counts)
    plt.xticks(xs, class_names, rotation=45, ha="right")
    plt.ylabel("# test samples")
    plt.title("Class support (test split)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_miscls_gallery(test_subset, y_true, y_pred, out_png, max_imgs=24, img_size=128):
    mis_idx = np.where(y_true != y_pred)[0].tolist()
    if not mis_idx:
        fig = plt.figure(figsize=(5, 3))
        plt.text(0.5, 0.5, "No misclassifications.", ha="center", va="center")
        plt.axis("off")
        fig.savefig(out_png, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return

    picks = mis_idx[:max_imgs]
    base_ds = test_subset.dataset  # ImageFolder
    idxs = test_subset.indices     # indices into base_ds.samples

    imgs = []
    for i in picks:
        base_idx = idxs[i]
        path, _ = base_ds.samples[base_idx]
        try:
            im = Image.open(path).convert("RGB").resize((img_size, img_size))
            imgs.append(transforms.ToTensor()(im))
        except Exception:
            continue

    if not imgs:
        return

    grid = make_grid(imgs, nrow=max(1, int(np.sqrt(len(imgs)))))
    nd = grid.permute(1, 2, 0).numpy()

    fig = plt.figure(figsize=(8, 8))
    plt.imshow(nd)
    plt.axis("off")
    plt.title("Misclassification gallery (first few)")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_tsne_features(model, test_dl, device, class_names, out_png):
    feat_extractor = nn.Sequential(*list(model.children())[:-1]).to(device).eval()
    feats, labels = [], []
    with torch.no_grad():
        for x, y in test_dl:
            x = x.to(device, non_blocking=True)
            f = feat_extractor(x)          # [B, 512, 1, 1]
            f = torch.flatten(f, 1).cpu()  # [B, 512]
            feats.append(f)
            labels.append(y.cpu())
    if not feats:
        return
    X = torch.cat(feats, dim=0).numpy()
    y = torch.cat(labels, dim=0).numpy()

    tsne = TSNE(n_components=2, perplexity=max(5, min(30, len(y)//10 or 5)),
                init="pca", learning_rate="auto", random_state=0)
    Z = tsne.fit_transform(X)

    fig = plt.figure(figsize=(7, 6))
    for c in range(len(class_names)):
        mask = (y == c)
        if np.any(mask):
            plt.scatter(Z[mask, 0], Z[mask, 1], s=12, alpha=0.7, label=class_names[c])
    plt.legend(markerscale=1.5, fontsize=8, ncol=2)
    plt.title("t-SNE of penultimate features")
    plt.xlabel("dim-1"); plt.ylabel("dim-2")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_loss_acc(history, out_png):
    ep = np.arange(1, len(history["train_loss"]) + 1)
    fig = plt.figure(figsize=(8, 6))
    plt.plot(ep, history["train_loss"], label="train_loss")
    plt.plot(ep, history["val_loss"], label="val_loss")
    plt.plot(ep, history["train_acc"], label="train_acc")
    plt.plot(ep, history["val_acc"], label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.title("Training curves")
    plt.legend()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------- dataloaders + model ----------
def get_loaders(
    data_root,
    img_size=224,
    bs=32,
    val_frac=0.15,
    test_frac=0.15,
    seed=42,
    num_workers=0,
):
    tf_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.20, hue=0.03),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    tf_eval = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    full_ds = datasets.ImageFolder(root=str(data_root), transform=tf_train)
    class_names = full_ds.classes
    N = len(full_ds)
    n_test = int(N * test_frac)
    n_val = int(N * val_frac)
    n_train = max(N - n_val - n_test, 1)

    set_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        full_ds, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(seed)
    )

    val_ds.dataset.transform = tf_eval
    test_ds.dataset.transform = tf_eval

    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_dl, val_dl, test_dl, class_names, train_ds, val_ds, test_ds


def build_model(n_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, n_classes)
    return model


def run_epoch(model, dl, criterion, optimizer=None, device="cuda"):
    model.train(mode=optimizer is not None)
    total_loss, total_correct, total = 0.0, 0, 0
    all_logits, all_targets = [], []

    for x, y in dl:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == y).sum().item()
            total += y.size(0)

        total_loss += loss.item() * y.size(0)
        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())

    avg_loss = total_loss / max(total, 1)
    acc = total_correct / max(total, 1)
    logits_cat = torch.cat(all_logits, dim=0).numpy() if all_logits else np.zeros((0, 1))
    targets_cat = torch.cat(all_targets, dim=0).numpy() if all_targets else np.zeros((0,), dtype=np.int64)
    return avg_loss, acc, logits_cat, targets_cat


# ---------- training script ----------
def main():
    # deterministic setup
    set_seed(CONFIG["SEED"])

    # resolve all paths relative to this file (robust to different CWDs)
    script_dir = Path(__file__).resolve().parent
    src_root = (script_dir / CONFIG["SRC_ROOT"]).resolve()
    bench = (script_dir / "benchmarking").resolve()
    ds_root = bench / "dataset_flat"
    out_dir = bench / f"resnet18_{CONFIG['VIEW']}"

    safe_mkdir(bench)
    safe_mkdir(ds_root)
    safe_mkdir(out_dir)

    # early guard: source root must exist and contain at least one class dir
    if not src_root.exists():
        # create a scaffold to guide the student, then exit gracefully
        (src_root / "open_palm" / CONFIG["VIEW"]).mkdir(parents=True, exist_ok=True)
        (src_root / "closed_fist" / CONFIG["VIEW"]).mkdir(parents=True, exist_ok=True)
        msg = (
            f"[setup] created scaffold at:\n"
            f"  {src_root}\\<class>\\{CONFIG['VIEW']}\\*.jpg\n"
            f"please place your images and re-run."
        )
        (out_dir / "SETUP_INSTRUCTIONS.txt").write_text(msg, encoding="utf-8")
        print(f"[ERROR] source dataset folder was missing: {src_root}")
        print(msg)
        return

    # 1) pack into ImageFolder layout
    classes, class_counts = build_imagefolder_from_dataset(
        src_root, ds_root, view=CONFIG["VIEW"], mode=CONFIG["PACK_MODE"]
    )
    if not classes:
        print("[ERROR] no classes packed. ensure you have:")
        print(f"  {src_root}\\<class>\\{CONFIG['VIEW']}\\*.jpg")
        return
    (out_dir / "labels.txt").write_text("\n".join(classes), encoding="utf-8")
    with open(out_dir / "class_counts.json", "w", encoding="utf-8") as f:
        json.dump(class_counts, f, indent=2)

    # 2) dataloaders + dataset handles (keep subsets for galleries)
    train_dl, val_dl, test_dl, class_names, train_ds, val_ds, test_ds = get_loaders(
        ds_root,
        img_size=CONFIG["IMG_SIZE"],
        bs=CONFIG["BATCH_SIZE"],
        val_frac=CONFIG["VAL_FRAC"],
        test_frac=CONFIG["TEST_FRAC"],
        seed=CONFIG["SEED"],
        num_workers=CONFIG["NUM_WORKERS"],
    )

    # 3) model/opt/sched
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(n_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["LR"], weight_decay=CONFIG["WD"])
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["EPOCHS"])

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = -1.0

    # 4) train
    for ep in range(1, CONFIG["EPOCHS"] + 1):
        tr_loss, tr_acc, _, _ = run_epoch(model, train_dl, criterion, optimizer, device)
        va_loss, va_acc, _, _ = run_epoch(model, val_dl,   criterion, None,      device)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        scheduler.step()

        print(f"Epoch {ep:02d}/{CONFIG['EPOCHS']} | "
              f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} acc={va_acc:.4f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), out_dir / "best_resnet18.pt")

    # 5) test on best checkpoint
    model.load_state_dict(torch.load(out_dir / "best_resnet18.pt", map_location=device))
    te_loss, te_acc, te_logits, te_targets = run_epoch(model, test_dl, criterion, None, device)
    print(f"[test] loss={te_loss:.4f} acc={te_acc:.4f}")

    # confusion matrices
    y_pred = np.argmax(te_logits, axis=1)
    cm = confusion_matrix(te_targets, y_pred, labels=list(range(len(class_names))))
    plot_confusion(cm, class_names, out_dir / "confusion_matrix.png", normalize=False)
    plot_confusion(cm, class_names, out_dir / "confusion_matrix_norm.png", normalize=True)

    # probabilities for PR/ROC/calibration/top-k
    y_prob = (torch.softmax(torch.tensor(te_logits), dim=1)).numpy() if te_logits.size else np.zeros((0, len(class_names)))

    if y_prob.shape[0] > 0:
        plot_pr_curves(te_targets, y_prob, class_names,
                       out_png=out_dir / "pr_curves.png",
                       out_auc_bars=out_dir / "pr_auc_bars.png")
        plot_roc_curves(te_targets, y_prob, class_names, out_png=out_dir / "roc_curves.png")
        plot_calibration(te_targets, y_prob, out_png=out_dir / "calibration_reliability.png", n_bins=10)
        plot_topk_curve(te_logits, te_targets, out_png=out_dir / "topk_curve.png",
                        k_max=min(5, y_prob.shape[1]))

    # class support histogram
    plot_class_support(te_targets, class_names, out_png=out_dir / "class_support.png")

    # misclassification gallery
    make_miscls_gallery(test_ds, te_targets, y_pred, out_png=out_dir / "miscls_gallery.png",
                        max_imgs=CONFIG["MISCLS_MAX"], img_size=128)

    # t-SNE over penultimate features
    plot_tsne_features(model, test_dl, device, class_names, out_png=out_dir / "tsne_features.png")

    # training curves
    plot_loss_acc(history, out_dir / "loss_acc_curves.png")

    # text + JSON reports
    report = classification_report(te_targets, y_pred, target_names=class_names, digits=4, zero_division=0)
    (out_dir / "report.txt").write_text(report, encoding="utf-8")
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "test_loss": float(te_loss),
                "test_acc": float(te_acc),
                "classes": class_names,
                "class_counts": class_counts,
            },
            f,
            indent=2,
        )

    # 6) export ONNX
    model.eval()
    dummy = torch.randn(1, 3, CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"], device=device)
    torch.onnx.export(
        model,
        dummy,
        out_dir / "best_resnet18.onnx",
        input_names=["input"],
        output_names=["logits"],
        opset_version=18,
        do_constant_folding=True,
    )

    print(f"\nSaved artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
