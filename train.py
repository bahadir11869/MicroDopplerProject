"""
Paper Section IV — CNN eğitim/değerlendirme.

Adımlar:
  1) Birden fazla σ_P değeri için sentetik dataset üret (Fig. 2 chain)
  2) %80/%10/%10 stratified split
  3) CNN'i Adam + CE loss ile eğit (Section III-B, Eq. 19-20)
  4) İki ayrı test:
       a) In-domain   → Fig. 2 sentetik held-out
       b) Cross-domain → Fig. 6 MTI'lı sentetik (paper'ın "measurement" testine analog)
  5) Confusion matrix (Eq. 21-22) + σ_P-accuracy eğrisi (Fig. 13)

Paper Fig. 13: σ_P=0.20 civarında accuracy zirvesi (gerçek ölçüm üzerinde test).
Sentetik held-out üzerinde monoton düşüş bekleriz; cross-domain (MTI test)
çan eğrisi vermeli — paper'ın domain-gap davranışına analog.
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

import main as paper
from cnn_model import MicroDopplerCNN

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["HELI", "QUAD"]


def make_dataset(n_per_class: int, sigma_p: float, seed: int, use_mti: bool = False,
                 measurement_proxy: bool = False) -> tuple:
    """Bellekte sentetik dataset üretir.

    measurement_proxy=True → paperın ölçüm davranışı (sabit σ_P=0.20 + AWGN).
    use_mti=True           → Fig. 6 MTI zinciri (yapısal farklı, domain gap testi).
    """
    rng = np.random.default_rng(seed)
    classes = [(paper.HELI_REF_PARAMS, 0), (paper.QUAD_REF_PARAMS, 1)]
    if measurement_proxy:
        gen = lambda p, sigma_p, rng: paper.generate_profile_measurement(p, rng=rng)
    elif use_mti:
        gen = paper.generate_profile_mti
    else:
        gen = paper.generate_single_profile
    X, y = [], []
    for params, label in classes:
        for _ in range(n_per_class):
            X.append(gen(params, sigma_p=sigma_p, rng=rng))
            y.append(label)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def make_loaders(X: np.ndarray, y: np.ndarray, batch_size: int) -> tuple:
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.20, random_state=SEED, stratify=y)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=SEED, stratify=y_tmp)

    def loader(Xs, ys, shuffle):
        Xt = torch.as_tensor(Xs, dtype=torch.float32).unsqueeze(1)
        yt = torch.as_tensor(ys, dtype=torch.long)
        return DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=shuffle,
                          pin_memory=DEVICE.type == "cuda")

    return loader(X_tr, y_tr, True), loader(X_va, y_va, False), loader(X_te, y_te, False)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, return_preds: bool = False):
    """Loss + accuracy (+ optional preds for CM)."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss, total_correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for Xb, yb in loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        logits = model(Xb)
        total_loss += criterion(logits, yb).item()
        preds = logits.argmax(1)
        total_correct += (preds == yb).sum().item()
        total += yb.size(0)
        if return_preds:
            all_preds.append(preds.cpu().numpy())
            all_labels.append(yb.cpu().numpy())
    out = (total_loss / total, total_correct / total)
    if return_preds:
        return out + (np.concatenate(all_preds), np.concatenate(all_labels))
    return out


def cross_domain_test(model: nn.Module, n_per_class: int, batch_size: int):
    """Eğitilmiş modeli ölçüm proxy'si üzerinde test eder (paper Fig. 13 analoğu).

    Ölçüm proxy SABİT σ_P=0.20 + AWGN içerir (paper Section IV-C-1: ölçüm
    setinin "true" σ_P'si ≈ 0.20). Eğitim σ_P'si bu değere yaklaştıkça
    accuracy tepe yapar. Yani burada geçilen `sigma_p` değil, modelin
    EĞİTİMDE kullandığı σ_P belirleyici.
    """
    X, y = make_dataset(n_per_class, sigma_p=0.0, seed=SEED + 9999,
                        measurement_proxy=True)
    Xt = torch.as_tensor(X, dtype=torch.float32).unsqueeze(1)
    yt = torch.as_tensor(y, dtype=torch.long)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=False)
    return evaluate(model, loader, return_preds=True)


def plot_confusion_matrix(cm: np.ndarray, title: str, fname: str):
    """Eq. 21-22 metrikleri ile birlikte CM görseli."""
    import matplotlib.pyplot as plt
    tp, fn = cm[1, 1], cm[1, 0]
    fp, tn = cm[0, 1], cm[0, 0]
    acc = (tp + tn) / cm.sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"{title}\nacc={acc:.3f}  precision(QUAD)={prec:.3f}")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(fname, dpi=140)
    plt.close()


def train_one_sigma(sigma_p: float, n_per_class: int, epochs: int, batch_size: int, lr: float,
                    do_cross_domain: bool = False, save_cm: bool = False) -> dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    t0 = time.time()
    X, y = make_dataset(n_per_class, sigma_p, seed=SEED + int(round(sigma_p * 1000)))
    tr_loader, va_loader, te_loader = make_loaders(X, y, batch_size)
    print(f"[σ_P={sigma_p:.2f}] dataset={X.shape}, üretim={time.time()-t0:.1f}s")

    model = MicroDopplerCNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for ep in range(1, epochs + 1):
        model.train()
        for Xb, yb in tr_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(Xb), yb)
            loss.backward()
            opt.step()
        val_loss, val_acc = evaluate(model, va_loader)
        best_val_acc = max(best_val_acc, val_acc)
        if ep == 1 or ep % max(1, epochs // 5) == 0 or ep == epochs:
            print(f"  ep {ep:3d}/{epochs}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

    # In-domain test (Fig. 2 held-out)
    test_loss, test_acc, te_preds, te_labels = evaluate(model, te_loader, return_preds=True)
    cm_in = confusion_matrix(te_labels, te_preds, labels=[0, 1])
    print(f"[σ_P={sigma_p:.2f}] ✅ in-domain test_acc={test_acc:.4f}")
    print(f"  CM (in-domain): TN={cm_in[0,0]} FP={cm_in[0,1]} | FN={cm_in[1,0]} TP={cm_in[1,1]}")

    result = {
        "sigma_p": sigma_p,
        "test_acc_indomain": test_acc,
        "val_acc": best_val_acc,
        "test_loss": test_loss,
        "cm_indomain": cm_in.tolist(),
    }

    # Cross-domain test (ölçüm proxy — paperın "measurement" analoğu, σ_P=0.20 sabit)
    if do_cross_domain:
        _, cd_acc, cd_preds, cd_labels = cross_domain_test(model, n_per_class // 4, batch_size)
        cm_cd = confusion_matrix(cd_labels, cd_preds, labels=[0, 1])
        print(f"[σ_P={sigma_p:.2f}] 🌉 cross-domain (MTI) test_acc={cd_acc:.4f}")
        print(f"  CM (cross-domain): TN={cm_cd[0,0]} FP={cm_cd[0,1]} | FN={cm_cd[1,0]} TP={cm_cd[1,1]}")
        result["test_acc_crossdomain"] = cd_acc
        result["cm_crossdomain"] = cm_cd.tolist()

    if save_cm:
        plot_confusion_matrix(cm_in, f"In-domain (σ_P={sigma_p:.2f})", f"cm_indomain_sp{sigma_p:.2f}.png")
        if do_cross_domain:
            plot_confusion_matrix(cm_cd, f"Cross-domain MTI (σ_P={sigma_p:.2f})",
                                  f"cm_crossdomain_sp{sigma_p:.2f}.png")

    return result


def plot_fig13(results: list, out: str = "my_fig13.png"):
    import matplotlib.pyplot as plt
    sigmas = [r["sigma_p"] for r in results]
    in_accs = [r["test_acc_indomain"] for r in results]
    plt.figure(figsize=(8, 5))
    plt.plot(sigmas, in_accs, "o-", color="tab:blue", label="In-domain (Fig. 2 sentetik held-out)")
    if "test_acc_crossdomain" in results[0]:
        cd_accs = [r["test_acc_crossdomain"] for r in results]
        plt.plot(sigmas, cd_accs, "s--", color="tab:red", label="Cross-domain (Fig. 6 MTI)")
    plt.xlabel("σ_P (unitless)")
    plt.ylabel("Accuracy")
    plt.title("Paper Fig. 13 tarzı — σ_P vs accuracy")
    plt.grid(True, alpha=0.3)
    plt.ylim(0.4, 1.02)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    print(f"✅ {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigmas", nargs="+", type=float,
                    default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60])
    ap.add_argument("--n_per_class", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--cross_domain", action="store_true", help="Fig. 6 MTI testi de çalıştır")
    ap.add_argument("--save_cm", action="store_true", help="her σ_P için CM resmi kaydet")
    ap.add_argument("--out", default="train_results.json")
    args = ap.parse_args()

    print(f"Device: {DEVICE}")
    results = []
    for s in args.sigmas:
        r = train_one_sigma(s, args.n_per_class, args.epochs, args.batch_size, args.lr,
                            do_cross_domain=args.cross_domain, save_cm=args.save_cm)
        results.append(r)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    plot_fig13(results)


if __name__ == "__main__":
    main()
