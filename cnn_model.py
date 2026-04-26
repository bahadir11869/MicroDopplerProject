"""
Paper Section III-B, Fig. 7 — Proposed CNN Architecture.

Akış:
  400×1 → reshape 1×20×20
  → Conv(1→32, K=3) + ReLU          → 32×18×18
  → MaxPool(2×2)                     → 32×9×9
  → Conv(32→64, K=3) + ReLU          → 64×7×7
  → AvgPool(7×7)                     → 64×1×1
  → Flatten                          → 64
  → FC(64→32) + ReLU
  → FC(32→2)
  → Softmax (cross-entropy loss içinde)

Adam optimizer, cross-entropy loss (Eq. 19-20).
"""
import torch
import torch.nn as nn


class MicroDopplerCNN(nn.Module):
    """Paper Fig. 7: iki conv + iki pool + iki FC."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        # Feature extraction
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)   # 20→18
        self.pool1 = nn.MaxPool2d(kernel_size=2)                                 # 18→9
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)   # 9→7
        self.pool2 = nn.AvgPool2d(kernel_size=7)                                 # 7→1
        self.relu = nn.ReLU(inplace=True)

        # Classifier
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Giriş: (N, 1, 400) — CNN için 1×20×20'ye reshape
        if x.dim() == 3:
            x = x.view(x.size(0), 1, 20, 20)
        elif x.dim() == 2:
            x = x.view(x.size(0), 1, 20, 20)

        x = self.relu(self.conv1(x))     # (N, 32, 18, 18)
        x = self.pool1(x)                # (N, 32, 9, 9)
        x = self.relu(self.conv2(x))     # (N, 64, 7, 7)
        x = self.pool2(x)                # (N, 64, 1, 1)
        x = torch.flatten(x, 1)          # (N, 64)
        x = self.relu(self.fc1(x))       # (N, 32)
        x = self.fc2(x)                  # (N, 2) — CE loss softmax'ı kendi uygular
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    net = MicroDopplerCNN()
    x = torch.randn(4, 1, 400)
    y = net(x)
    print(f"Giriş: {tuple(x.shape)}")
    print(f"Çıkış: {tuple(y.shape)}")
    print(f"Toplam parametre: {count_parameters(net):,}")
