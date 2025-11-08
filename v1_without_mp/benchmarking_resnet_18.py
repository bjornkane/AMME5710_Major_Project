# benchmark_from_dataset_resnet18.py
# Walks: dataset/<class>/{images_crop|images_raw|masks|overlays}/*
# Packs into: benchmarking/dataset_flat/<class>/*.jpg  (copy/link/move)
# Trains ResNet-18 and saves all artifacts into: benchmarking/resnet18_<view>/
#
# Usage examples (run from the project root that contains 'dataset/'):
#   python benchmark_from_dataset_resnet18.py --view images_crop --epochs 20 --bs 32 --lr 1e-3
#   python benchmark_from_dataset_resnet18.py --view images_raw --link
#
import argparse, os, json, random, shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve

# ---------------- utilities ----------------
def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def is_image(p: Path):
    return p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"}

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def copy_or_link(src: Path, dst: Path, mode: str):
    if mode == "move":
        shutil.move(str(src), str(dst))
    elif mode == "link":
        try:
            os.link(src, dst)  # hard link on same volume
        except Exception:
            shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)

# ---------------- packer ----------------
def build_imagefolder_from_dataset(src_root: Path, dst_root: Path, view: str, mode="copy"):
    """
    Expects:
      src_root/
        <classA>/{images_crop|images_raw|masks|overlays}/*
        <classB>/{...}
    Copies/links images from chosen 'view' into:
      dst_root/<class>/*.jpg
    Returns: (sorted classes list, class_counts dict)
    """
    safe_mkdir(dst_root)
    class_counts = {}
    classes = []
    for cls_dir in sorted([p for p in src_root.iterdir() if p.is_dir()]):
        cls = cls_dir.name
        src_view = cls_dir / view
        if not src_view.exists():
            # fallback: prefer images_crop, then images_raw
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
        print(f"[pack] {cls:<12} view={src_view.name:<12} -> {n:5d} files")
    classes = sorted(classes)
    return classes, class_counts

# ---------------- plots ----------------
def plot_confusion(cm, class_names, out_png):
    fig = plt.figure(figsize=(8,7))
    im = plt.imshow(cm, interpolation='nearest', cmap="Blues")
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("count")
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    thresh = cm.max()/2.0 if cm.size>0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout(); plt.ylabel("true"); plt.xlabel("pred")
    fig.savefig(out_png, dpi=180, bbox_inches="tight"); plt.close(fig)

def plot_pr_curves(y_true, y_prob, class_names, out_png):
    fig = plt.figure(figsize=(8,7))
    C = y_prob.shape[1]
    for c in range(C):
        y_bin = (y_true == c).astype(np.uint8)
        precision, recall, _ = precision_recall_curve(y_bin, y_prob[:, c])
        plt.plot(recall, precision, label=class_names[c])
    plt.xlabel("recall"); plt.ylabel("precision"); plt.title("per-class PR")
    plt.legend(loc="lower left", ncol=2, fontsize=8)
    fig.savefig(out_png, dpi=180, bbox_inches="tight"); plt.close(fig)

def plot_loss_acc(history, out_png):
    ep = np.arange(1, len(history["train_loss"])+1)
    fig = plt.figure(figsize=(8,6))
    plt.plot(ep, history["train_loss"], label="train_loss")
    plt.plot(ep, history["val_loss"],   label="val_loss")
    plt.plot(ep, history["train_acc"],  label="train_acc")
    plt.plot(ep, history["val_acc"],    label="val_acc")
    plt.xlabel("epoch"); plt.ylabel("value"); plt.title("training curves"); plt.legend()
    fig.savefig(out_png, dpi=180, bbox_inches="tight"); plt.close(fig)

# ---------------- data + model ----------------
def get_loaders(data_root, img_size=224, bs=32, val_frac=0.15, test_frac=0.15, seed=42):
    tf_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.20, hue=0.03),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
    ])
    tf_eval = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
    ])
    full_ds = datasets.ImageFolder(root=str(data_root), transform=tf_train)
    class_names = full_ds.classes
    N = len(full_ds)
    n_test = int(N*test_frac)
    n_val  = int(N*val_frac)
    n_train = max(N - n_val - n_test, 1)
    set_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        full_ds, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(seed)
    )
    val_ds.dataset.transform = tf_eval
    test_ds.dataset.transform = tf_eval
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
    return train_dl, val_dl, test_dl, class_names

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
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == y).sum().item()
            total += y.size(0)
        total_loss += loss.item() * y.size(0)
        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())
    avg_loss = total_loss / max(total, 1)
    acc = total_correct / max(total, 1)
    logits_cat = torch.cat(all_logits, dim=0).numpy() if all_logits else np.zeros((0,1))
    targets_cat = torch.cat(all_targets, dim=0).numpy() if all_targets else np.zeros((0,), dtype=np.int64)
    return avg_loss, acc, logits_cat, targets_cat

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, default="dataset", help="root that contains class folders")
    ap.add_argument("--view", type=str, default="images_crop",
                    choices=["images_crop","images_raw","masks","overlays"])
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--move", action="store_true", help="move files into dataset_flat")
    ap.add_argument("--link", action="store_true", help="hard-link if possible, else copy")
    args = ap.parse_args()

    set_seed(args.seed)
    root = Path(".").resolve()
    src_root = (root / args.src).resolve()
    bench = root / "benchmarking"
    ds_root = bench / "dataset_flat"
    out_dir = bench / f"resnet18_{args.view}"
    safe_mkdir(bench); safe_mkdir(ds_root); safe_mkdir(out_dir)

    mode = "copy"
    if args.move: mode = "move"
    if args.link: mode = "link"

    # 1) pack
    classes, class_counts = build_imagefolder_from_dataset(src_root, ds_root, view=args.view, mode=mode)
    if not classes:
        print("No classes packed. Check folder structure under 'dataset/'.")
        return
    (out_dir / "labels.txt").write_text("\n".join(classes), encoding="utf-8")
    with open(out_dir / "class_counts.json", "w", encoding="utf-8") as f:
        json.dump(class_counts, f, indent=2)

    # 2) loaders
    train_dl, val_dl, test_dl, class_names = get_loaders(ds_root, img_size=args.img, bs=args.bs)

    # 3) train
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(n_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}
    best_val_acc = -1.0

    for ep in range(1, args.epochs+1):
        tr_loss, tr_acc, _ , _ = run_epoch(model, train_dl, criterion, optimizer, device)
        va_loss, va_acc, _ , _ = run_epoch(model, val_dl,   criterion, None,      device)
        history["train_loss"].append(tr_loss); history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc);   history["val_acc"].append(va_acc)
        scheduler.step()
        print(f"Epoch {ep:02d}/{args.epochs} | train_loss={tr_loss:.4f} acc={tr_acc:.4f} | val_loss={va_loss:.4f} acc={va_acc:.4f}")
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), out_dir/"best_resnet18.pt")

    # 4) test + reports
    model.load_state_dict(torch.load(out_dir/"best_resnet18.pt", map_location=device))
    te_loss, te_acc, te_logits, te_targets = run_epoch(model, test_dl, criterion, None, device)
    print(f"[test] loss={te_loss:.4f} acc={te_acc:.4f}")

    y_pred = np.argmax(te_logits, axis=1)
    cm = confusion_matrix(te_targets, y_pred, labels=list(range(len(class_names))))
    plot_confusion(cm, class_names, out_dir/"confusion_matrix.png")

    y_prob = (torch.softmax(torch.tensor(te_logits), dim=1)).numpy() if te_logits.size else np.zeros((0, len(class_names)))
    if y_prob.shape[0] > 0:
        plot_pr_curves(te_targets, y_prob, class_names, out_dir/"pr_curves.png")

    # loss/acc curves
    plot_loss_acc(history, out_dir/"loss_acc_curves.png")

    report = classification_report(te_targets, y_pred, target_names=class_names, digits=4, zero_division=0)
    (out_dir/"report.txt").write_text(report, encoding="utf-8")
    with open(out_dir/"summary.json","w",encoding="utf-8") as f:
        json.dump({
            "test_loss": float(te_loss),
            "test_acc": float(te_acc),
            "classes": class_names,
            "class_counts": class_counts
        }, f, indent=2)

    # ONNX
    model.eval(); dummy = torch.randn(1,3,224,224, device=("cuda" if torch.cuda.is_available() else "cpu"))
    torch.onnx.export(
        model, dummy, out_dir/"best_resnet18.onnx",
        input_names=["input"], output_names=["logits"],
        opset_version=18, do_constant_folding=True
    )

    print(f"\nSaved artifacts to: {out_dir}")

if __name__ == "__main__":
    main()
