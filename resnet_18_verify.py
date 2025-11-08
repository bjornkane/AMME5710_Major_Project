# resnet_18_verify.py
#
# Student intent:
#   Reload artifacts from benchmarking/resnet18_<view>/ and cross-verify:
#     - test accuracy matches summary.json (within tolerance)
#     - confusion matrix sums to #test samples
#     - top-k curve is non-decreasing
#     - calibration ECE is in [0,1]
#     - dataset_flat file counts match class_counts.json
#     - optional ONNX parity: PyTorch vs ONNX logits "close"
#
# Usage:
#   python resnet_18_verify.py
#
# Notes:
#   - No CLI flags; edit CONFIG if your output folder name changes.
#   - ONNX parity now adapts to the model's expected batch dimension.

import json, random, hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve

# ---------------- CONFIG (edit if needed) ----------------
CONFIG = dict(
    VIEW_DIR="resnet18_images_crop",  # folder inside benchmarking/
    IMG_SIZE=224,
    BATCH_SIZE=32,
    SEED=42,
    NUM_WORKERS=0,                    # Windows-safe default
)
# ---------------------------------------------------------


# ---------- small helpers ----------
def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def sha256_file(p: Path):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_model(n_classes: int) -> nn.Module:
    # same classifier head shape as trainer
    m = models.resnet18(weights=None)
    in_feats = m.fc.in_features
    m.fc = nn.Linear(in_feats, n_classes)
    return m


def get_loaders_from_flat(ds_root: Path, img_size=224, bs=32, seed=42, num_workers=0):
    # deterministic transforms (no augment) to match evaluation
    tf_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])
    tf_eval = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])

    full_ds = datasets.ImageFolder(root=str(ds_root), transform=tf_train)
    class_names = full_ds.classes
    N = len(full_ds)
    n_test = int(N * 0.15)
    n_val  = int(N * 0.15)
    n_train = max(N - n_val - n_test, 1)

    set_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        full_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed)
    )
    # switch to eval transforms for val/test
    val_ds.dataset.transform  = tf_eval
    test_ds.dataset.transform = tf_eval

    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_dl, val_dl, test_dl, class_names, test_ds


def run_epoch_logits(model, dl, device="cpu"):
    # forward entire loader, return logits/targets for metric checks
    model.eval()
    crit = nn.CrossEntropyLoss()
    total_loss, total_correct, total = 0.0, 0, 0
    all_logits, all_targets = [], []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = crit(logits, y)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == y).sum().item()
            total += y.size(0)
            total_loss += loss.item() * y.size(0)
            all_logits.append(logits.cpu())
            all_targets.append(y.cpu())
    avg_loss = total_loss / max(total, 1)
    acc = total_correct / max(total, 1)
    logits_cat  = torch.cat(all_logits,  dim=0).numpy() if all_logits  else np.zeros((0,1))
    targets_cat = torch.cat(all_targets, dim=0).numpy() if all_targets else np.zeros((0,), dtype=np.int64)
    return avg_loss, acc, logits_cat, targets_cat


def try_onnx_parity(onnx_path: Path, pt_model: nn.Module, img_size=224, device="cpu"):
    # compare PyTorch vs ONNX logits; adapt batch to ONNX input shape
    try:
        import onnxruntime as ort
    except Exception:
        return False, "onnxruntime not installed — skipping ONNX parity."

    if not onnx_path.exists():
        return False, "ONNX file not found — skipping ONNX parity."

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    inp_name = inp.name
    # decide batch size to feed:
    # - if first dim is an int and ==1 → feed 1
    # - if None/dynamic → feed 4 for a stronger check
    # - else fallback to 1 for safety
    batch_dim = inp.shape[0]
    if isinstance(batch_dim, int):
        B = max(1, min(1, batch_dim))  # if 1 → 1; if other int → keep 1 for safety
    else:
        B = 4  # dynamic axes → can test >1

    dummy = torch.randn(B, 3, img_size, img_size, device=device)
    with torch.no_grad():
        pt_logits = pt_model(dummy).cpu().numpy().astype(np.float32)

    ort_inputs = {inp_name: dummy.cpu().numpy().astype(np.float32)}
    ort_logits = sess.run([sess.get_outputs()[0].name], ort_inputs)[0].astype(np.float32)

    # shapes must match
    if pt_logits.shape != ort_logits.shape:
        return False, f"PyTorch logits shape {pt_logits.shape} != ONNX logits shape {ort_logits.shape}"

    # numerical closeness (allow small export/runtime tolerances)
    close = np.allclose(pt_logits, ort_logits, rtol=1e-3, atol=1e-2)
    msg = "PyTorch vs ONNX logits close." if close else "WARNING: PyTorch vs ONNX differ."
    return close, msg


def main():
    set_seed(CONFIG["SEED"])
    script_dir = Path(__file__).resolve().parent
    bench = script_dir / "benchmarking"
    out_dir = bench / CONFIG["VIEW_DIR"]
    ds_root = bench / "dataset_flat"

    # required artifacts
    must = [
        out_dir / "best_resnet18.pt",
        out_dir / "summary.json",
        out_dir / "class_counts.json",
        out_dir / "labels.txt",
    ]
    for p in must:
        if not p.exists():
            print(f"[FAIL] missing artifact: {p}")
            return

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    class_counts_saved = json.loads((out_dir / "class_counts.json").read_text(encoding="utf-8"))
    labels = (out_dir / "labels.txt").read_text(encoding="utf-8").strip().splitlines()
    n_classes = len(labels)

    # check dataset_flat counts vs saved counts
    class_counts_actual = {}
    for cls in labels:
        cls_dir = ds_root / cls
        class_counts_actual[cls] = sum(
            1 for p in cls_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"}
        )
    counts_match = (class_counts_actual == class_counts_saved)
    print(f"[check] class_counts match: {counts_match}")
    if not counts_match:
        print("  saved :", class_counts_saved)
        print("  actual:", class_counts_actual)

    # rebuild deterministic splits
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, test_dl, class_names, test_subset = get_loaders_from_flat(
        ds_root,
        img_size=CONFIG["IMG_SIZE"],
        bs=CONFIG["BATCH_SIZE"],
        seed=CONFIG["SEED"],
        num_workers=CONFIG["NUM_WORKERS"],
    )
    assert class_names == labels, "[FAIL] ImageFolder class order != labels.txt order."

    # load model and re-test
    model = build_model(n_classes)
    model.load_state_dict(torch.load(out_dir / "best_resnet18.pt", map_location=device))
    model = model.to(device).eval()

    te_loss, te_acc, te_logits, te_targets = run_epoch_logits(model, test_dl, device=device)
    print(f"[retest] acc={te_acc:.4f} (summary says {summary.get('test_acc', None)})")
    acc_ok = (abs(te_acc - float(summary["test_acc"])) <= 1e-3)
    print(f"[check] accuracy matches summary: {acc_ok}")

    # confusion matrix sum = #test samples
    y_pred = np.argmax(te_logits, axis=1)
    cm = confusion_matrix(te_targets, y_pred, labels=list(range(n_classes)))
    cm_sum_ok = (cm.sum() == len(te_targets))
    print(f"[check] confusion matrix sum OK: {cm_sum_ok} ({cm.sum()} vs {len(te_targets)})")

    # top-k monotonicity (k=1..min(5,C))
    ks = list(range(1, min(5, n_classes) + 1))
    accs = []
    for k in ks:
        topk = torch.topk(torch.tensor(te_logits), k=k, dim=1).indices.numpy()
        hits = np.any(topk == te_targets[:, None], axis=1).mean() if len(te_targets) else 0.0
        accs.append(hits)
    topk_ok = all(accs[i] <= accs[i+1] + 1e-12 for i in range(len(accs)-1))
    print(f"[check] top-k monotonic: {topk_ok}  values={['%.3f'%a for a in accs]}")

    # calibration ECE in [0,1]
    if te_logits.size:
        y_prob = torch.softmax(torch.tensor(te_logits), dim=1).numpy()
        conf = y_prob.max(axis=1)
        correct = (te_targets == np.argmax(y_prob, axis=1)).astype(np.uint8)
        prob_true, prob_pred = calibration_curve(correct, conf, n_bins=10, strategy="uniform")
        ece = float(np.mean(np.abs(prob_pred - prob_true)))
        ece_ok = (0.0 - 1e-6 <= ece <= 1.0 + 1e-6)
        print(f"[check] calibration ECE bounds: {ece_ok} (ECE={ece:.3f})")
    else:
        ece_ok = True

    # ONNX parity (now batch-aware)
    onnx_ok, onnx_msg = try_onnx_parity(out_dir / "best_resnet18.onnx", model, img_size=CONFIG["IMG_SIZE"], device=device)
    print(f"[onnx ] {onnx_msg}")

    # record small checksums for reproducibility
    ck_files = ["best_resnet18.pt", "best_resnet18.onnx", "report.txt", "summary.json"]
    checksums = {}
    for name in ck_files:
        p = out_dir / name
        if p.exists(): checksums[name] = sha256_file(p)

    # write summary report
    lines = [
        f"class_counts_match={counts_match}",
        f"acc_match={acc_ok}",
        f"cm_sum_ok={cm_sum_ok}",
        f"topk_monotonic={topk_ok}",
        f"ece_ok={ece_ok}",
        f"onnx_ok={onnx_ok}",
        "checksums=" + json.dumps(checksums, indent=2),
    ]
    (out_dir / "verification_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n[done] wrote verification_report.txt")

if __name__ == "__main__":
    main()
