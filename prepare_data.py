import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

SEED = 42


def visualize_samples(X, y):
    """HELI (0) ve QUAD (1) için birer örnek profil çiz — makale Şekil 11 karşılaştırması."""
    idx_heli = int(np.where(y == 0)[0][0])
    idx_quad = int(np.where(y == 1)[0][0])

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(X[idx_heli], color="blue")
    plt.title("HELI Sentetik Menzil Profili")
    plt.xlabel("Örnek İndeksi (0-400)")
    plt.ylabel("Normalize Genlik")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(X[idx_quad], color="red")
    plt.title("QUAD Sentetik Menzil Profili")
    plt.xlabel("Örnek İndeksi (0-400)")
    plt.ylabel("Normalize Genlik")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def prepare_dataloaders(X_path="X_data.npy", y_path="y_labels.npy",
                        batch_size=128, visualize=True, num_workers=0):
    """
    Makaleyle aynı bölünme: %80 / %10 / %10 (train / val / test), stratified.
    CNN girişi: (N, C=1, L=400).
    """
    if not (os.path.exists(X_path) and os.path.exists(y_path)):
        raise FileNotFoundError(f"{X_path} veya {y_path} bulunamadı. Önce main.py çalıştırın.")

    X = np.load(X_path)
    y = np.load(y_path)
    print(f"Yüklenen veri: X={X.shape}, y={y.shape}")

    if visualize:
        visualize_samples(X, y)

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=SEED, stratify=y_tmp
    )
    print(f"Train={X_train.shape[0]}  Val={X_val.shape[0]}  Test={X_test.shape[0]}")

    def to_tensor(arr, dtype):
        return torch.as_tensor(arr, dtype=dtype)

    def make_loader(Xs, ys, shuffle):
        Xt = to_tensor(Xs, torch.float32).unsqueeze(1)  # (N, 1, 400)
        yt = to_tensor(ys, torch.long)
        ds = TensorDataset(Xt, yt)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=torch.cuda.is_available())

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader = make_loader(X_val, y_val, shuffle=False)
    test_loader = make_loader(X_test, y_test, shuffle=False)

    print(f"✅ DataLoader'lar hazır (batch_size={batch_size}).")
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    prepare_dataloaders()
