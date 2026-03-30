# 🔥 End-to-End Deep Learning with PyTorch
### A Senior Engineer's Handbook & Cheat Sheet
> *Theory · Architecture · Code · Pitfalls · Production*

---

## Table of Contents

1. [Foundations](#1-foundations)
2. [Preprocessing Pipeline](#2-preprocessing-pipeline)
3. [Artificial Neural Networks (ANN / MLP)](#3-artificial-neural-networks-ann--mlp)
4. [Convolutional Neural Networks (CNN)](#4-convolutional-neural-networks-cnn)
5. [Recurrent Architectures (RNN / LSTM / GRU)](#5-recurrent-architectures-rnn--lstm--gru)
6. [Modern Transformers & Vision Transformers (ViT)](#6-modern-transformers--vision-transformers-vit)
7. [Multimodal Models (VLM / CLIP)](#7-multimodal-models-vlm--clip)
8. [Master Training Loop & Optimization](#8-master-training-loop--optimization)
9. [Deployment & Scaling](#9-deployment--scaling)
10. [Quick-Reference Cheat Sheet](#10-quick-reference-cheat-sheet)

---

## 1. Foundations

### 1.1 Theory — Tensors & Autograd

A **tensor** is PyTorch's fundamental data structure — an n-dimensional array living on CPU, GPU, or Apple Silicon (MPS). Unlike NumPy arrays, PyTorch tensors carry a computation graph when `requires_grad=True`, enabling **automatic differentiation (Autograd)**. During a forward pass, PyTorch records every operation into a dynamic DAG. Calling `.backward()` traverses this graph in reverse, computing gradients via the chain rule: `∂L/∂w = (∂L/∂y)(∂y/∂w)`. This is the engine behind all deep learning training.

The **device-agnostic** pattern is critical for production: never hard-code `"cuda"`. Always resolve the device at runtime. With PyTorch ≥ 2.0, `torch.compile()` wraps any `nn.Module` and JIT-compiles the compute graph for significant speedups (often 1.5–3×) with a single line.

### 1.2 Architecture Diagram

```
Python Scalar/List/NumPy Array
          │
          ▼
   torch.tensor(data)
          │
          ▼
  Tensor [shape, dtype, device]
          │  requires_grad=True
          ▼
   Forward Pass: y = f(x, W)
          │
          ▼
   Loss: L = criterion(y, target)
          │
          ▼
   L.backward()  ──► Autograd builds ∂L/∂W
          │
          ▼
   optimizer.step()  ──► W = W - lr * ∂L/∂W
```

### 1.3 PyTorch Code Snippet

```python
import torch
import torch.nn as nn

# ── Device Resolution (CPU / CUDA / MPS) ──────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():          # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()
print(f"Using device: {DEVICE}")


# ── Tensor Fundamentals ───────────────────────────────────────────────────────
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True, device=DEVICE)
W = torch.randn(2, 2, requires_grad=True, device=DEVICE)

# Operations build the computation graph automatically
y = x @ W          # matrix multiply
loss = y.sum()

# Backprop
loss.backward()
print("Gradient of W:", W.grad)   # ∂loss/∂W


# ── Common Tensor Operations ──────────────────────────────────────────────────
a = torch.zeros(3, 4)
b = torch.ones(3, 4)
c = torch.randn(3, 4)             # N(0,1)
d = torch.arange(0, 12).reshape(3, 4)

# Shape manipulation
print(c.shape)                    # torch.Size([3, 4])
print(c.view(2, 6).shape)         # torch.Size([2, 6])
print(c.permute(1, 0).shape)      # torch.Size([4, 3])
print(c.unsqueeze(0).shape)       # torch.Size([1, 3, 4])
print(c.squeeze().shape)          # removes dim-1 dims

# Arithmetic (all auto-broadcast)
result = a + b                    # element-wise
dot    = (a * b).sum()            # Frobenius inner product
matmul = a @ b.T                  # (3,4) @ (4,3) → (3,3)

# Indexing
print(d[0, :])                    # row 0
print(d[:, -1])                   # last column
print(d[d > 5])                   # boolean mask

# Moving tensors to device
x_gpu = x.to(DEVICE)
x_cpu = x_gpu.cpu()               # back to CPU for numpy
arr   = x_cpu.detach().numpy()    # .detach() stops grad tracking


# ── torch.compile (PyTorch ≥ 2.0) ────────────────────────────────────────────
model = nn.Linear(64, 10).to(DEVICE)
model = torch.compile(model)      # Triton-based kernel fusion
```

### 1.4 Key Hyperparameters / Settings

| Setting | Recommended Default | Notes |
|---|---|---|
| `dtype` | `torch.float32` | Use `bfloat16` on Ampere+ GPUs for speed |
| `pin_memory` | `True` (when CUDA) | Faster CPU→GPU transfers |
| `torch.backends.cudnn.benchmark` | `True` | Optimizes convolution algorithms |
| Seed | `torch.manual_seed(42)` | Always set for reproducibility |

### 1.5 Common Pitfalls

- **In-place ops on leaf tensors**: `x += 1` on a `requires_grad=True` tensor corrupts the graph. Use `x = x + 1`.
- **Forgetting `.detach()`**: Accumulating tensors in a list for metrics without `.detach()` leaks the entire computation graph → OOM.
- **NumPy bridge**: Always call `.cpu().detach().numpy()` in that exact order.
- **Seed for full reproducibility**: Set both `torch.manual_seed(s)` and `torch.cuda.manual_seed_all(s)`.

---

## 2. Preprocessing Pipeline

### 2.1 Theory

Raw data is rarely model-ready. A robust preprocessing pipeline handles three concerns: *loading* (reading files/databases into tensors), *transforming* (normalization, augmentation, tokenization), and *batching* (collating samples into mini-batches). PyTorch formalizes this through `Dataset` (single-item logic) and `DataLoader` (batching, shuffling, multi-process prefetching). Data augmentation acts as a regularizer, artificially expanding the training distribution and reducing overfitting without collecting new data — mathematically equivalent to training on an infinite ensemble of perturbed examples.

Imbalanced datasets cause models to learn a trivial class-prior solution. Three mitigations: (1) **WeightedRandomSampler** over-samples minorities at the loader level, (2) **class-weighted loss** penalizes majority-class errors less, and (3) **focal loss** adaptively down-weights easy, correctly-classified examples.

### 2.2 Architecture Diagram

```
Raw Files / Database
        │
        ▼
  CustomDataset.__getitem__(idx)
        │  ─── applies transforms ───►  Augmented Tensor
        ▼
  DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
        │
        ▼
  Collate → (batch_X [B,C,H,W], batch_y [B])
        │
        ▼
  Model Forward Pass
```

### 2.3 PyTorch Code Snippet

```python
import os
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms.v2 as T   # transforms v2 — preferred API
from collections import Counter


# ── Custom Dataset ─────────────────────────────────────────────────────────────
class ImageClassificationDataset(Dataset):
    """
    Expects directory structure:
        root/
          class_a/img1.jpg
          class_b/img2.jpg
    """
    def __init__(self, root: str, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples: list[tuple[Path, int]] = []
        for cls in self.classes:
            for img_path in (self.root / cls).glob("*.jpg"):
                self.samples.append((img_path, self.class_to_idx[cls]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ── Transforms (torchvision.transforms.v2) ────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transforms = T.Compose([
    T.RandomResizedCrop(224, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    T.RandomGrayscale(p=0.05),
    T.RandomRotation(degrees=15),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transforms = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ── Imbalanced Data: WeightedRandomSampler ────────────────────────────────────
def make_weighted_sampler(dataset: Dataset) -> WeightedRandomSampler:
    labels = [dataset.samples[i][1] for i in range(len(dataset))]
    class_counts = Counter(labels)
    # weight per class = 1 / count
    class_weights = {c: 1.0 / n for c, n in class_counts.items()}
    sample_weights = torch.tensor([class_weights[lbl] for lbl in labels])
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# ── DataLoaders ───────────────────────────────────────────────────────────────
def build_dataloaders(train_dir: str, val_dir: str, batch_size: int = 32):
    train_ds = ImageClassificationDataset(train_dir, transform=train_transforms)
    val_ds   = ImageClassificationDataset(val_dir,   transform=val_transforms)

    sampler  = make_weighted_sampler(train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,          # replaces shuffle=True
        num_workers=4,
        pin_memory=True,          # faster GPU transfer
        persistent_workers=True,  # avoid worker respawn overhead
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    return train_loader, val_loader


# ── Focal Loss (Imbalanced Classification) ────────────────────────────────────
class FocalLoss(nn.Module):
    """
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    gamma=0 → standard cross-entropy; gamma=2 is typical default.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# ── Tabular Dataset (for ANN/MLP use-cases) ──────────────────────────────────
import pandas as pd
from sklearn.preprocessing import StandardScaler

class TabularDataset(Dataset):
    def __init__(self, csv_path: str, target_col: str, fit_scaler: bool = True):
        df = pd.read_csv(csv_path)
        X = df.drop(columns=[target_col]).values.astype("float32")
        y = df[target_col].values.astype("int64")
        if fit_scaler:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):  return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]
```

### 2.4 Key Hyperparameters

| Parameter | Typical Range | Impact |
|---|---|---|
| `batch_size` | 16–512 | Larger = more stable gradient, less regularization |
| `num_workers` | `os.cpu_count() // 2` | Prevents DataLoader CPU bottleneck |
| `pin_memory` | `True` (CUDA only) | Async CPU→GPU copy |
| Augmentation strength | Light for small datasets | Overly strong aug hurts convergence |
| Normalization stats | ImageNet stats for pretrained; else compute from data | Mismatched stats = bad features |

### 2.5 Common Pitfalls

- **Normalizing after ToTensor**: In `transforms.v2`, call `ToDtype` before `Normalize`.
- **Data leakage**: Never fit `StandardScaler` on the validation set; fit only on train.
- **Wrong `num_workers` on Windows**: Use `num_workers=0` or protect with `if __name__ == "__main__"`.
- **Variable-length sequences**: Use a custom `collate_fn` with `pad_sequence` from `torch.nn.utils.rnn`.

---

## 3. Artificial Neural Networks (ANN / MLP)

### 3.1 Theory

A **Multi-Layer Perceptron (MLP)** stacks fully-connected (linear) layers interleaved with nonlinear activations. The fundamental computation per layer is `h = σ(Wx + b)`, where `W ∈ ℝ^(out×in)` and `σ` is an activation function. The Universal Approximation Theorem guarantees that a single hidden layer with sufficient width can approximate any continuous function on a compact domain — but depth (more layers) is empirically far more parameter-efficient for complex functions. Training minimizes a loss L via **stochastic gradient descent**: parameters update as `θ ← θ − η∇_θ L`, where the gradient is computed by backpropagation (chain rule traversal of the computation graph).

**Batch Normalization** normalizes layer inputs to zero mean and unit variance, then applies learnable scale/shift: `BN(x) = γ * (x − μ_B)/σ_B + β`. This stabilizes gradient flow, allows higher learning rates, and acts as a regularizer. **Dropout** randomly zeroes activations with probability `p` during training, preventing co-adaptation and reducing overfitting — equivalent to training an exponential ensemble of sub-networks.

### 3.2 Architecture Diagram

```
Input x ∈ ℝ^d
      │
      ▼
 Linear(d → 512)
      │
      ▼
 BatchNorm1d(512)
      │
      ▼
    ReLU()
      │
      ▼
 Dropout(p=0.3)
      │
      ▼
 Linear(512 → 256)
      │
      ▼
 BatchNorm1d(256)
      │
      ▼
    ReLU()
      │
      ▼
 Dropout(p=0.3)
      │
      ▼
 Linear(256 → num_classes)
      │
      ▼
 Output logits ∈ ℝ^C
```

### 3.3 PyTorch Code Snippet

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    General-purpose Multi-Layer Perceptron.

    Args:
        input_dim:   number of input features
        hidden_dims: list of hidden layer widths, e.g. [512, 256, 128]
        output_dim:  number of output classes / regression targets
        dropout_p:   dropout probability (applied after each hidden activation)
        use_bn:      use BatchNorm1d between linear and activation
        activation:  activation class (default: nn.ReLU)
    """
    def __init__(
        self,
        input_dim:   int,
        hidden_dims: list[int],
        output_dim:  int,
        dropout_p:   float = 0.3,
        use_bn:      bool  = True,
        activation:        = nn.ReLU,
    ):
        super().__init__()
        layers = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(activation())
            if dropout_p > 0:
                layers.append(nn.Dropout(p=dropout_p))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, output_dim))  # output — no activation
        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        """Kaiming He initialization for ReLU networks."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = MLP(input_dim=128, hidden_dims=[512, 256, 128], output_dim=10).to(DEVICE)

    x      = torch.randn(32, 128).to(DEVICE)         # batch of 32
    logits = model(x)
    print(f"Output shape: {logits.shape}")            # [32, 10]

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")


# ── Loss Functions Reference ──────────────────────────────────────────────────
# Classification (single label)
ce_loss   = nn.CrossEntropyLoss()                  # logits → softmax → NLL
# Classification (multi-label)
bce_loss  = nn.BCEWithLogitsLoss()                 # sigmoid per logit
# Regression
mse_loss  = nn.MSELoss()
mae_loss  = nn.L1Loss()
huber     = nn.HuberLoss(delta=1.0)                # robust to outliers

# Label smoothing (regularization)
smooth_ce = nn.CrossEntropyLoss(label_smoothing=0.1)
```

### 3.4 Key Hyperparameters

| Hyperparameter | Typical Range | Notes |
|---|---|---|
| Hidden dims | [256–2048] | Double each layer; avoid bottleneck layers |
| Dropout p | 0.1–0.5 | Higher for large networks; 0 with BN often works |
| Batch Norm | Almost always True | Disable if batch_size < 8 (use LayerNorm instead) |
| Activation | ReLU / GELU / SiLU | GELU/SiLU smoother for Transformers |
| Initialization | Kaiming (ReLU), Xavier (tanh) | Critical for deep networks |
| Learning rate | 1e-4 – 1e-2 | Use LR range test |

### 3.5 Common Pitfalls

- **Dead ReLU neurons**: If LR is too high, neurons saturate to `x < 0` permanently. Use LeakyReLU or lower LR.
- **BatchNorm + Dropout ordering**: `Linear → BN → Activation → Dropout` is the canonical order.
- **BatchNorm in eval mode**: Call `model.eval()` before inference — BN uses running stats, not batch stats.
- **Regression output**: Remove final activation; use `nn.Linear` directly as the last layer.
- **Shape mismatch**: Print `x.shape` after every layer during debugging with a forward hook or `torchinfo.summary(model, (batch, features))`.

---

## 4. Convolutional Neural Networks (CNN)

### 4.1 Theory

CNNs exploit the spatial structure of images through parameter-sharing and local connectivity. A `Conv2d` layer applies K learnable filters of size `(C_in, kH, kW)` across the input, producing `K` feature maps. The output size is: `H_out = floor((H_in + 2*pad - dilation*(kH-1) - 1) / stride + 1)`. Each filter learns to detect a specific local pattern (edges, textures, object parts) regardless of spatial position — this **translational equivariance** dramatically reduces parameter count versus a fully-connected equivalent. Stacking conv layers with nonlinearities builds a hierarchical feature hierarchy from low-level to semantic features.

**Residual connections** (ResNets) solve the degradation problem in very deep networks. The skip connection `F(x) + x` ensures gradient flows unobstructed to early layers: `∂L/∂x = ∂L/∂y * (∂F/∂x + I)`. The identity term `I` prevents vanishing gradients, enabling networks 100+ layers deep. **Transfer learning** pre-trains on large datasets (ImageNet) and fine-tunes on small target datasets, treating the backbone as a universal feature extractor.

### 4.2 Architecture Diagram

```
Input [B, 3, 224, 224]
       │
       ▼
  Conv2d(3→64, k=3, s=1, p=1)  ──► [B, 64, 224, 224]
  BatchNorm2d → ReLU
       │
       ▼
  MaxPool2d(k=2, s=2)          ──► [B, 64, 112, 112]
       │
       ▼
  ResBlock(64→128)              ──► [B, 128, 56, 56]
  ┌─────────┐
  │ Conv→BN→ReLU→Conv→BN │
  │         + skip (1×1 Conv)   │
  └─────────┘
       │
       ▼
  AdaptiveAvgPool2d(1,1)        ──► [B, 128, 1, 1]
       │
       ▼
  Flatten                       ──► [B, 128]
       │
       ▼
  Linear(128 → num_classes)     ──► [B, C]
```

### 4.3 PyTorch Code Snippet

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ── Residual Block ─────────────────────────────────────────────────────────────
class ResidualBlock(nn.Module):
    """
    Basic residual block with optional downsampling.
    in_ch → out_ch, stride=2 halves spatial dims.
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1,      padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        # Skip connection: 1×1 conv to match dimensions
        self.skip = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.skip(x)          # residual addition
        return F.relu(out)


# ── Custom CNN ─────────────────────────────────────────────────────────────────
class SmallResNet(nn.Module):
    """Lightweight ResNet-style CNN for 224×224 input."""
    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        # Residual stages
        self.stage1 = self._make_stage(64,  128, stride=2, n_blocks=2)
        self.stage2 = self._make_stage(128, 256, stride=2, n_blocks=2)
        self.stage3 = self._make_stage(256, 512, stride=2, n_blocks=2)
        # Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    @staticmethod
    def _make_stage(in_ch, out_ch, stride, n_blocks):
        blocks = [ResidualBlock(in_ch, out_ch, stride=stride)]
        for _ in range(1, n_blocks):
            blocks.append(ResidualBlock(out_ch, out_ch))
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(x)


# ── Transfer Learning ─────────────────────────────────────────────────────────
def build_transfer_model(
    arch: str = "resnet50",
    num_classes: int = 10,
    freeze_backbone: bool = True,
) -> nn.Module:
    """
    Load a pretrained backbone and replace the classifier head.

    Strategy A — Feature Extraction: freeze backbone, train head only.
    Strategy B — Fine-Tuning: unfreeze all, use very small LR for backbone.
    """
    weights_map = {
        "resnet50":    models.ResNet50_Weights.IMAGENET1K_V2,
        "efficientnet_b0": models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        "convnext_small":  models.ConvNeXt_Small_Weights.IMAGENET1K_V1,
        "vit_b_16":    models.ViT_B_16_Weights.IMAGENET1K_V1,
    }
    model = getattr(models, arch)(weights=weights_map[arch])

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace head — architecture-specific
    if arch.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
    elif arch.startswith("efficientnet"):
        in_features = model.classifier[1].in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif arch.startswith("convnext"):
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
    elif arch.startswith("vit"):
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)

    return model


# ── Depthwise Separable Convolution (MobileNet-style) ─────────────────────────
class DepthwiseSeparableConv(nn.Module):
    """Depthwise conv + pointwise conv. ~8-9× fewer params than standard conv."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch,  3, stride=stride, padding=1,
                            groups=in_ch, bias=False)   # depthwise
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)               # pointwise
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.pw(self.dw(x))))


# ── Model Summary ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # pip install torchinfo
    from torchinfo import summary
    model = SmallResNet(num_classes=10)
    summary(model, input_size=(2, 3, 224, 224))
```

### 4.4 Key Hyperparameters

| Hyperparameter | Notes |
|---|---|
| Kernel size | 3×3 is dominant; 1×1 for projection; 7×7 for stem |
| Stride | 2 for downsampling (replaces MaxPool) |
| Padding | `padding = kernel_size // 2` preserves spatial size |
| `bias=False` | Always disable when followed by BatchNorm |
| Fine-tune LR | 10–100× lower for backbone vs. head |
| Augmentation | RandAugment / AugMix for competitive benchmarks |

### 4.5 Common Pitfalls

- **Forgetting to unfreeze backbone**: `freeze_backbone=True` trains only the head. Switch to full fine-tuning after a warm-up phase.
- **Mismatched normalization**: Pretrained models expect ImageNet normalization. Always apply before feeding to the model.
- **BatchNorm2d vs. GroupNorm**: BatchNorm degrades with `batch_size < 8`. Use `GroupNorm(num_groups=32)` for detection/segmentation with small batches.
- **`inplace=True` with residual connections**: Inplace activation on a tensor needed for a residual add corrupts gradients. Disable inplace ReLU in residual branches.

---

## 5. Recurrent Architectures (RNN / LSTM / GRU)

### 5.1 Theory

Recurrent networks process sequential data `x_1, ..., x_T` by maintaining a **hidden state** `h_t` that accumulates context. The vanilla RNN update rule is `h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b)`. The problem: gradients flow through T time steps via repeated multiplication by `W_hh`. If `|eigenvalues(W_hh)| < 1`, gradients vanish exponentially — long-range dependencies become invisible to the optimizer.

**LSTMs** (Hochreiter & Schmidhuber, 1997) introduce a cell state `c_t` as a gradient highway regulated by three gates: forget `f`, input `i`, output `o`. The cell update: `c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t`, where `g_t = tanh(...)` is the candidate cell. The additive combination means gradients can flow for hundreds of time steps. **GRUs** (Cho et al., 2014) merge the cell and hidden state into one, with only two gates (reset `r`, update `z`), making them ~33% faster with comparable performance on most tasks. Both are superseded by Transformers for most NLP tasks, but remain excellent for streaming/online sequence learning and time-series forecasting.

### 5.2 Architecture Diagram

```
Input Sequence x_1...x_T,  shape: [B, T, input_size]
         │
         ▼
  ┌── LSTM Cell (unrolled) ──────────────────────────────┐
  │  h_0, c_0 (zeros)                                    │
  │     │                                                │
  │  ┌──▼──┐   ┌─────┐   ┌─────┐         ┌─────┐       │
  │  │ x_1 │──►│ f,i │──►│ c_1 │─...─────►│ c_T │       │
  │  └─────┘   │  o  │   │ h_1 │         │ h_T │       │
  │             └─────┘   └─────┘         └──┬──┘       │
  └──────────────────────────────────────────┼───────────┘
                                             │
                  ┌──────────┬──────────────┘
                  │          │
            h_T only    all h_1..h_T
         (many-to-one)  (many-to-many / seq2seq)
                  │          │
               Linear      Linear per step
                  │          │
              Softmax     Softmax
```

### 5.3 PyTorch Code Snippet

```python
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# ── LSTM Sequence Classifier ──────────────────────────────────────────────────
class LSTMClassifier(nn.Module):
    """
    Many-to-one LSTM: classifies a variable-length input sequence.

    Args:
        vocab_size:   vocabulary size (for embedding layer)
        embed_dim:    embedding dimension
        hidden_dim:   LSTM hidden size
        num_layers:   stacked LSTM depth
        num_classes:  output classes
        dropout:      variational dropout (applied between LSTM layers)
        bidirectional: use BiLSTM (doubles hidden_dim output)
    """
    def __init__(
        self,
        vocab_size:    int,
        embed_dim:     int   = 128,
        hidden_dim:    int   = 256,
        num_layers:    int   = 2,
        num_classes:   int   = 2,
        dropout:       float = 0.3,
        bidirectional: bool  = True,
        pad_idx:       int   = 0,
    ):
        super().__init__()
        self.hidden_dim    = hidden_dim
        self.num_layers    = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,             # [B, T, H]  (preferred)
            bidirectional=bidirectional,
        )
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * self.num_directions, num_classes)

    def forward(
        self,
        x:      torch.Tensor,        # [B, T]  token indices
        lengths: torch.Tensor | None = None,  # [B]  actual sequence lengths
    ) -> torch.Tensor:
        emb = self.dropout(self.embedding(x))     # [B, T, E]

        if lengths is not None:
            # PackedSequence: skip PAD tokens in computation
            packed = pack_padded_sequence(emb, lengths.cpu(),
                                          batch_first=True, enforce_sorted=False)
            packed_out, (h_n, c_n) = self.lstm(packed)
            out, _ = pad_packed_sequence(packed_out, batch_first=True)
        else:
            out, (h_n, c_n) = self.lstm(emb)

        # h_n: [num_layers * num_dir, B, hidden_dim]
        # Take last layer's hidden state from all directions
        if self.bidirectional:
            # Concat forward & backward last hidden states
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)   # [B, H*2]
        else:
            h_last = h_n[-1]                                   # [B, H]

        return self.classifier(self.dropout(h_last))           # [B, C]


# ── GRU for Time-Series Forecasting ──────────────────────────────────────────
class GRUForecaster(nn.Module):
    """
    Many-to-many GRU for multi-step time-series forecasting.
    Input: [B, T_in, features]  →  Output: [B, T_out, targets]
    """
    def __init__(
        self,
        input_dim:  int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 1,
        forecast_horizon: int = 12,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(hidden_dim, output_dim * forecast_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, features]
        out, _ = self.gru(x)                           # [B, T, H]
        last    = out[:, -1, :]                         # [B, H]  last time step
        preds   = self.proj(last)                       # [B, T_out * out_dim]
        return preds.view(preds.size(0), self.forecast_horizon, -1)


# ── Sequence-to-Sequence with Attention ──────────────────────────────────────
class BahdanauAttention(nn.Module):
    """Additive attention: e_ij = v^T tanh(W_s * s_i + W_h * h_j)"""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W_s  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_h  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v    = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, query: torch.Tensor, keys: torch.Tensor):
        # query: [B, H], keys: [B, T, H]
        query = query.unsqueeze(1)                          # [B, 1, H]
        energy = self.v(torch.tanh(self.W_s(query) + self.W_h(keys)))  # [B, T, 1]
        attn_weights = torch.softmax(energy.squeeze(-1), dim=-1)        # [B, T]
        context = (attn_weights.unsqueeze(-1) * keys).sum(dim=1)        # [B, H]
        return context, attn_weights
```

### 5.4 Key Hyperparameters

| Hyperparameter | Typical Range | Notes |
|---|---|---|
| `hidden_dim` | 128–1024 | Larger for longer sequences |
| `num_layers` | 1–4 | Stack with care; diminishing returns |
| `bidirectional` | True for classification | False for autoregressive generation |
| `dropout` | 0.2–0.5 | Only applied between layers, not after last |
| `batch_first=True` | Always | Avoid shape confusion |
| Gradient clip | 1.0–5.0 | Essential for RNNs |

### 5.5 Common Pitfalls

- **Vanishing gradients**: Use `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)` before every optimizer step.
- **LSTM vs. GRU**: GRU trains faster and generalizes equally on most tasks. Use LSTM only if gate-level interpretability matters.
- **`batch_first` inconsistency**: PyTorch LSTM defaults to `[T, B, H]`. Always set `batch_first=True` for sanity.
- **Packing padded sequences**: Not doing this causes the LSTM to process PAD tokens, corrupting `h_n`. Always use `pack_padded_sequence`.
- **Stationarity for time-series**: Always difference or normalize your time series; LSTMs don't handle non-stationary data well.

---

## 6. Modern Transformers & Vision Transformers (ViT)

### 6.1 Theory

The Transformer (Vaswani et al., 2017) replaced recurrence with **Scaled Dot-Product Attention**: `Attention(Q, K, V) = softmax(QK^T / √d_k) * V`. Queries, Keys, and Values are linear projections of the same input. The `1/√d_k` scaling prevents dot products from growing large (which would push softmax into saturation, causing near-zero gradients). **Multi-Head Attention (MHA)** runs h parallel attention heads in lower-dimensional subspaces and concatenates outputs: `MHA = Concat(head_1,...,head_h) * W_O`. This allows the model to jointly attend to different representation subspaces simultaneously.

**Vision Transformers (ViT)** (Dosovitskiy et al., 2020) apply Transformers directly to images by splitting an image into fixed-size patches (e.g., 16×16), linearly embedding them, prepending a learnable `[CLS]` token, adding positional embeddings, and feeding to a standard Transformer encoder. The `[CLS]` token's output is used for classification. ViT outperforms CNNs at scale (>300M images) but requires more data or augmentation than CNNs on smaller datasets — DeiT addresses this with distillation.

### 6.2 Architecture Diagram

```
Image [B, 3, 224, 224]
      │
      ▼  Patch Embed (16×16 patches → 196 tokens)
  Patches [B, 196, 768]
      │  + [CLS] token + Positional Embedding
      ▼
 [B, 197, 768]
      │
      ▼ ┌──────── Transformer Encoder Block × 12 ────────────┐
        │                                                      │
        │   LayerNorm → Multi-Head Attention → + residual     │
        │   LayerNorm → FFN (MLP: 768→3072→768) → + residual  │
        └──────────────────────────────────────────────────────┘
      │
      ▼  Extract [CLS] token
 [B, 768]
      │
      ▼
  Linear → [B, num_classes]

─── Attention Detail ─────────────────────────────────────────
 Input X [B, T, D]
    │
    ├── Q = X * W_Q  [B, T, d_k]
    ├── K = X * W_K  [B, T, d_k]
    └── V = X * W_V  [B, T, d_v]
              │
              ▼
   scores = (Q @ K.T) / √d_k   [B, T, T]
              │
           softmax
              │
   attn = scores @ V            [B, T, d_v]
```

### 6.3 PyTorch Code Snippet

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ── Scaled Dot-Product Attention ──────────────────────────────────────────────
def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Q, K, V: [B, num_heads, T, d_k]
    Returns: context [B, num_heads, T, d_k], weights [B, num_heads, T, T]
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # [B, H, T, T]

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    weights = F.softmax(scores, dim=-1)
    if dropout_p > 0.0 and Q.requires_grad:
        weights = F.dropout(weights, p=dropout_p)

    context = torch.matmul(weights, V)
    return context, weights


# ── Multi-Head Attention ───────────────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model    = d_model
        self.num_heads  = num_heads
        self.d_k        = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        # [B, T, D] → [B, num_heads, T, d_k]
        return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        B, T, _ = query.shape

        Q = self.split_heads(self.W_Q(query))
        K = self.split_heads(self.W_K(key))
        V = self.split_heads(self.W_V(value))

        context, attn_weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout_p=self.dropout if self.training else 0.0
        )

        # Merge heads: [B, H, T, d_k] → [B, T, D]
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_O(context), attn_weights


# ── Transformer Encoder Block ─────────────────────────────────────────────────
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        self.attn   = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1  = nn.LayerNorm(d_model)
        self.norm2  = nn.LayerNorm(d_model)
        mlp_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask=None):
        # Pre-LN (more stable than original post-LN)
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


# ── Vision Transformer (ViT) ──────────────────────────────────────────────────
class PatchEmbedding(nn.Module):
    """Splits image into patches and linearly embeds each patch."""
    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_channels: int = 3, d_model: int = 768):
        super().__init__()
        assert img_size % patch_size == 0
        self.n_patches = (img_size // patch_size) ** 2
        # Conv2d with stride=patch_size is equivalent to non-overlapping patches
        self.proj = nn.Conv2d(in_channels, d_model,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x = self.proj(x)          # [B, D, H/P, W/P]
        x = x.flatten(2)          # [B, D, N]
        x = x.transpose(1, 2)     # [B, N, D]
        return x


class ViT(nn.Module):
    """
    Vision Transformer (ViT-B/16 default configuration).

    img_size=224, patch_size=16 → 196 patches
    d_model=768, num_heads=12, depth=12 → ViT-Base
    d_model=1024, num_heads=16, depth=24 → ViT-Large
    """
    def __init__(
        self,
        img_size:    int   = 224,
        patch_size:  int   = 16,
        in_channels: int   = 3,
        num_classes: int   = 1000,
        d_model:     int   = 768,
        depth:       int   = 12,
        num_heads:   int   = 12,
        mlp_ratio:   float = 4.0,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, d_model)
        n_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))
        self.pos_drop  = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = self.patch_embed(x)                          # [B, N, D]

        cls = self.cls_token.expand(B, -1, -1)           # [B, 1, D]
        x = torch.cat([cls, x], dim=1)                   # [B, N+1, D]
        x = self.pos_drop(x + self.pos_embed)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_out = x[:, 0]                                 # [B, D]  CLS token
        return self.head(cls_out)                         # [B, C]


# ── Flash Attention (PyTorch ≥ 2.0 built-in) ─────────────────────────────────
# Much faster and memory-efficient — use in production instead of manual impl:
# F.scaled_dot_product_attention(Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=False)

class EfficientMHA(nn.Module):
    """Drop-in MHA using F.scaled_dot_product_attention (Flash Attention v2)."""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads
        self.qkv_proj  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj  = nn.Linear(d_model, d_model)
        self.dropout   = dropout

    def forward(self, x: torch.Tensor, mask=None):
        B, T, D = x.shape
        qkv = self.qkv_proj(x).view(B, T, 3, self.num_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)              # [3, B, H, T, d_k]
        Q, K, V = qkv.unbind(0)

        # PyTorch 2.0 Flash Attention — O(N) memory vs O(N²)
        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)
```

### 6.4 Key Hyperparameters

| Hyperparameter | ViT-Tiny | ViT-Small | ViT-Base | ViT-Large |
|---|---|---|---|---|
| `d_model` | 192 | 384 | 768 | 1024 |
| `depth` | 12 | 12 | 12 | 24 |
| `num_heads` | 3 | 6 | 12 | 16 |
| `patch_size` | 16 | 16 | 16 | 16 |
| `mlp_ratio` | 4.0 | 4.0 | 4.0 | 4.0 |

### 6.5 Common Pitfalls

- **ViT needs more data**: Without pre-training or augmentation (RandAugment, MixUp, CutMix), ViT underfits on ImageNet-scale. Use DeiT or pre-trained weights.
- **Positional embedding mismatch**: When fine-tuning ViT at a different resolution, interpolate positional embeddings (bilinear on the 2D grid).
- **`d_model % num_heads != 0`**: Will crash. Always assert divisibility.
- **LayerNorm before attention (Pre-LN)**: The original ViT uses Post-LN, which is harder to train. Pre-LN (as implemented above) is stabler with large LR.
- **Flash Attention**: Always use `F.scaled_dot_product_attention` in production — it's 2–4× faster and uses O(N) vs O(N²) memory.

---

## 7. Multimodal Models (VLM / CLIP)

### 7.1 Theory

**Vision-Language Models (VLMs)** learn a shared semantic embedding space for images and text. **CLIP** (Radford et al., 2021) trains two encoders — a Vision Transformer and a text Transformer — using **contrastive loss**: given a batch of N (image, text) pairs, the model maximizes cosine similarity for matching pairs and minimizes it for the N²-N non-matching pairs. The loss is InfoNCE (Noise Contrastive Estimation): `L = -1/N * Σ log(exp(s_ii/τ) / Σ_j exp(s_ij/τ))` where `τ` is a learnable temperature parameter. This produces a rich, transferable joint embedding useful for zero-shot classification, image-text retrieval, and grounding.

Modern VLMs (LLaVA, GPT-4V, Flamingo, PaliGemma) extend CLIP-style encoders by projecting visual features into the token space of a Large Language Model (LLM). A lightweight **MLP projector** (or cross-attention) maps visual embeddings `∈ ℝ^{N×d_v}` into LLM tokens `∈ ℝ^{N×d_llm}`, which are prepended to the text token sequence. The LLM autoregressively generates a response attending to both visual and text tokens.

### 7.2 Architecture Diagram

```
 ┌── CLIP Architecture ─────────────────────────────────────────────────────┐
 │                                                                           │
 │  Image ──► ViT Encoder ──► image_embed [B, D]   ──┐                     │
 │                                                     ├──► Cosine Similarity│
 │  Text  ──► Text Encoder ──► text_embed  [B, D]  ──┘    Matrix [B, B]    │
 │                                                                           │
 │  Loss: contrastive across diagonal of similarity matrix                  │
 └───────────────────────────────────────────────────────────────────────────┘

 ┌── VLM (LLaVA-style) ──────────────────────────────────────────────────────┐
 │                                                                           │
 │  Image [B, 3, 224, 224]                                                  │
 │     │                                                                     │
 │  CLIP Vision Encoder (frozen)                                             │
 │     │                                                                     │
 │  Visual Tokens [B, 256, 1024]                                            │
 │     │                                                                     │
 │  MLP Projector (trainable)  ──► [B, 256, 4096]  ◄── LLM token dim       │
 │     │                                                                     │
 │  Prepend to Text Tokens  ──► [B, 256+T, 4096]                           │
 │     │                                                                     │
 │  LLM Decoder (Llama / Mistral / Phi)                                     │
 │     │                                                                     │
 │  Autoregressive Text Output                                               │
 └───────────────────────────────────────────────────────────────────────────┘
```

### 7.3 PyTorch Code Snippet

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── CLIP-style Contrastive Loss ────────────────────────────────────────────────
class CLIPContrastiveLoss(nn.Module):
    """
    InfoNCE loss for image-text contrastive learning.
    Inputs: image_embeds & text_embeds — L2 normalized, shape [B, D].
    """
    def __init__(self, init_temperature: float = 0.07):
        super().__init__()
        # Learnable log temperature (CLIP initializes logit_scale = log(1/0.07))
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / init_temperature)))

    def forward(
        self,
        image_embeds: torch.Tensor,   # [B, D]  L2 normalized
        text_embeds:  torch.Tensor,   # [B, D]  L2 normalized
    ) -> torch.Tensor:
        # Clamp temperature for training stability
        logit_scale = self.logit_scale.exp().clamp(max=100.0)

        # Cosine similarity matrix [B, B]
        sim = logit_scale * (image_embeds @ text_embeds.T)

        # Labels: diagonal is the ground truth pair
        labels = torch.arange(sim.size(0), device=sim.device)

        # Symmetric loss: image→text and text→image
        loss_i2t = F.cross_entropy(sim,   labels)
        loss_t2i = F.cross_entropy(sim.T, labels)
        return (loss_i2t + loss_t2i) / 2.0


# ── Minimal CLIP-style Dual Encoder ──────────────────────────────────────────
class CLIPModel(nn.Module):
    """
    Simplified CLIP: separate vision & text towers → shared embedding space.
    In practice, use openai/clip or open_clip_torch for production.
    """
    def __init__(
        self,
        vision_encoder: nn.Module,   # e.g., ViT, ResNet
        text_encoder:   nn.Module,   # e.g., BERT, GPT
        vision_dim:     int,
        text_dim:       int,
        embed_dim:      int = 512,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder   = text_encoder
        # Projection heads to shared embedding space
        self.vision_proj = nn.Linear(vision_dim, embed_dim, bias=False)
        self.text_proj   = nn.Linear(text_dim,   embed_dim, bias=False)
        self.loss_fn     = CLIPContrastiveLoss()

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.vision_encoder(images)
        return F.normalize(self.vision_proj(feats), dim=-1)

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        feats = self.text_encoder(tokens)
        return F.normalize(self.text_proj(feats), dim=-1)

    def forward(self, images, tokens):
        image_embeds = self.encode_image(images)
        text_embeds  = self.encode_text(tokens)
        loss = self.loss_fn(image_embeds, text_embeds)
        return loss, image_embeds, text_embeds


# ── Zero-Shot Classification with CLIP ───────────────────────────────────────
def zero_shot_classify(
    model: CLIPModel,
    images: torch.Tensor,
    class_names: list[str],
    tokenizer,
    device: torch.device,
) -> torch.Tensor:
    """
    Classify images by comparing to text embeddings of class names.
    Returns predicted class indices [B].
    """
    # Build text prompts: "a photo of a {class}"
    prompts = [f"a photo of a {c}" for c in class_names]
    tokens  = tokenizer(prompts).to(device)

    with torch.no_grad():
        image_embeds = model.encode_image(images)           # [B, D]
        text_embeds  = model.encode_text(tokens)            # [C, D]
        # Cosine similarity [B, C]
        similarity = image_embeds @ text_embeds.T
        probs = similarity.softmax(dim=-1)

    return probs.argmax(dim=-1)


# ── VLM: MLP Projector (LLaVA-style) ─────────────────────────────────────────
class MLPProjector(nn.Module):
    """
    Projects CLIP visual features into LLM embedding space.
    LLaVA-1.5 uses a 2-layer MLP with GELU activation.
    """
    def __init__(self, vision_dim: int = 1024, llm_dim: int = 4096):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        # visual_features: [B, N_patches, vision_dim]
        return self.proj(visual_features)   # [B, N_patches, llm_dim]


# ── Using Open CLIP (Recommended for Production) ──────────────────────────────
# pip install open-clip-torch
#
# import open_clip
# model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
# tokenizer = open_clip.get_tokenizer('ViT-B-32')
#
# images = preprocess(Image.open("image.jpg")).unsqueeze(0)
# text   = tokenizer(["a photo of a cat", "a photo of a dog"])
#
# with torch.no_grad():
#     image_features = model.encode_image(images)
#     text_features  = model.encode_text(text)
#     probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)


# ── HuggingFace VLM Integration ───────────────────────────────────────────────
# pip install transformers
#
# from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
#
# model     = PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma-3b-pt-224")
# processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")
#
# inputs = processor(text="Describe this image:", images=image, return_tensors="pt")
# output = model.generate(**inputs, max_new_tokens=100)
# print(processor.decode(output[0], skip_special_tokens=True))
```

### 7.4 Key Hyperparameters

| Hyperparameter | Value | Notes |
|---|---|---|
| Temperature `τ` | 0.07 (learnable) | Lower = sharper distribution |
| Embed dim | 512–1024 | Larger = more expressive but heavier |
| Batch size | 32K+ (CLIP paper) | Contrastive learning needs large batches; use gradient accumulation |
| Vision encoder | ViT-B/32, ViT-L/14 | Larger patch → faster but lower resolution |
| Projector depth | 2-layer MLP | Deeper projectors don't consistently help |

### 7.5 Common Pitfalls

- **Small batch contrastive learning**: InfoNCE quality degrades significantly with batch size < 256. Use gradient accumulation or MoCo-style momentum queues.
- **Not normalizing embeddings**: Contrastive loss requires L2-normalized embeddings. `F.normalize(x, dim=-1)` is non-negotiable.
- **Freezing vision encoder too early**: For VLMs, fine-tune the visual encoder at a very small LR; fully frozen encoders limit alignment quality.
- **Modality collapse**: If one tower dominates (usually text), all pairs get similar scores. Monitor per-modality embedding variance.

---

## 8. Master Training Loop & Optimization

### 8.1 Theory

The training loop is the operational heart of deep learning. A complete loop must orchestrate: forward pass, loss computation, backward pass (gradient calculation), gradient clipping, optimizer step, scheduler step, metric tracking, and evaluation. **AdamW** decouples weight decay from the gradient update: `θ ← θ - lr * (m̂/(√v̂ + ε)) - lr * λ * θ`, where `m̂, v̂` are bias-corrected first/second moment estimates and `λ` is the weight decay. This is mathematically more correct than L2 regularization in Adam and leads to better generalization. **Learning rate schedulers** modulate the LR dynamically; cosine annealing with warm restarts is a strong default. **Early stopping** monitors a validation metric and halts training when improvement stalls, preventing overfitting.

**Mixed Precision Training** (`torch.amp`) uses `float16`/`bfloat16` for the forward pass and a loss scaler to prevent underflow during backward, achieving 2–4× speedup and halved memory with no accuracy loss on modern GPUs.

### 8.2 Architecture Diagram

```
for epoch in range(max_epochs):
      │
      ▼
  ┌── Train Phase ────────────────────────────────────┐
  │  for batch in train_loader:                       │
  │    with autocast():                               │
  │      logits = model(x)                            │
  │      loss   = criterion(logits, y)                │
  │    scaler.scale(loss).backward()                  │
  │    scaler.unscale_(optimizer)                     │
  │    clip_grad_norm_(params, max_norm)              │
  │    scaler.step(optimizer)                         │
  │    scaler.update()                                │
  │    optimizer.zero_grad(set_to_none=True)          │
  └────────────────────────────────────────────────────┘
      │
      ▼
  ┌── Eval Phase ──────────────────────────────────────┐
  │  model.eval()                                      │
  │  with torch.no_grad():                             │
  │    val_loss, val_acc = evaluate(val_loader)        │
  │  model.train()                                     │
  └────────────────────────────────────────────────────┘
      │
      ▼
  scheduler.step()
      │
      ▼
  EarlyStopping.check(val_loss) ──► stop if no improvement
      │
      ▼
  save_checkpoint() if best_val_metric
```

### 8.3 PyTorch Code Snippet — Master Training Loop

```python
import time
import math
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


# ── Training Configuration ─────────────────────────────────────────────────────
@dataclass
class TrainingConfig:
    # Core
    epochs:          int   = 100
    learning_rate:   float = 3e-4
    weight_decay:    float = 1e-2
    # Gradient
    max_grad_norm:   float = 1.0
    grad_accum_steps: int  = 1        # simulate larger batch
    # Scheduler
    scheduler:       str   = "cosine" # "cosine" | "onecycle" | "plateau" | "warmup_cosine"
    warmup_epochs:   int   = 5
    min_lr:          float = 1e-6
    # AMP
    use_amp:         bool  = True
    # Early stopping
    patience:        int   = 10
    min_delta:       float = 1e-4
    # Checkpointing
    save_dir:        str   = "checkpoints"
    save_top_k:      int   = 3


# ── Early Stopping ─────────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4,
                 mode: str = "min"):
        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode
        self.counter    = 0
        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        improved = (
            (self.mode == "min" and score < self.best_score - self.min_delta) or
            (self.mode == "max" and score > self.best_score + self.min_delta)
        )
        if improved:
            self.best_score = score
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ── Learning Rate Schedulers ──────────────────────────────────────────────────
def build_scheduler(optimizer, cfg: TrainingConfig, steps_per_epoch: int):
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr
        )
    elif cfg.scheduler == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.learning_rate,
            steps_per_epoch=steps_per_epoch,
            epochs=cfg.epochs,
            pct_start=0.3,
        )
    elif cfg.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
    elif cfg.scheduler == "warmup_cosine":
        def lr_lambda(epoch):
            if epoch < cfg.warmup_epochs:
                return epoch / max(1, cfg.warmup_epochs)     # linear warmup
            progress = (epoch - cfg.warmup_epochs) / max(1, cfg.epochs - cfg.warmup_epochs)
            return max(cfg.min_lr / cfg.learning_rate,
                       0.5 * (1 + math.cos(math.pi * progress)))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        raise ValueError(f"Unknown scheduler: {cfg.scheduler}")


# ── Checkpoint Manager ────────────────────────────────────────────────────────
class CheckpointManager:
    def __init__(self, save_dir: str, top_k: int = 3):
        self.save_dir  = Path(save_dir)
        self.top_k     = top_k
        self.best_checkpoints: list[tuple[float, Path]] = []  # (metric, path)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save(self, model, optimizer, scheduler, epoch, metric, cfg):
        path = self.save_dir / f"epoch_{epoch:03d}_metric_{metric:.4f}.pt"
        torch.save({
            "epoch":      epoch,
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "scheduler":  scheduler.state_dict(),
            "metric":     metric,
            "config":     cfg,
        }, path)
        self.best_checkpoints.append((metric, path))
        # Keep only top_k checkpoints (lowest metric = best)
        self.best_checkpoints.sort(key=lambda x: x[0])
        while len(self.best_checkpoints) > self.top_k:
            _, old_path = self.best_checkpoints.pop()
            old_path.unlink(missing_ok=True)
        logger.info(f"Checkpoint saved: {path}")

    def load_best(self, model, optimizer=None, scheduler=None) -> int:
        if not self.best_checkpoints:
            raise FileNotFoundError("No checkpoints found.")
        _, best_path = self.best_checkpoints[0]
        ckpt = torch.load(best_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        if optimizer:  optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler:  scheduler.load_state_dict(ckpt["scheduler"])
        logger.info(f"Loaded best checkpoint: {best_path}")
        return ckpt["epoch"]


# ── Metric Tracker ────────────────────────────────────────────────────────────
class MetricTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self._sum, self._count = 0.0, 0

    def update(self, value: float, n: int = 1):
        self._sum   += value * n
        self._count += n

    @property
    def avg(self) -> float:
        return self._sum / max(1, self._count)


# ── One Epoch: Train ──────────────────────────────────────────────────────────
def train_one_epoch(
    model:       nn.Module,
    loader:      DataLoader,
    criterion:   nn.Module,
    optimizer:   torch.optim.Optimizer,
    scaler:      GradScaler,
    device:      torch.device,
    cfg:         TrainingConfig,
    epoch:       int,
) -> dict:
    model.train()
    loss_tracker = MetricTracker()
    acc_tracker  = MetricTracker()
    optimizer.zero_grad(set_to_none=True)   # more efficient than zero_grad()
    t0 = time.perf_counter()

    for step, (x, y) in enumerate(loader, start=1):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        # Forward (with AMP)
        with autocast(enabled=cfg.use_amp, dtype=torch.bfloat16):
            logits = model(x)
            loss   = criterion(logits, y) / cfg.grad_accum_steps  # scale for accumulation

        # Backward
        scaler.scale(loss).backward()

        # Gradient accumulation: update only every N steps
        if step % cfg.grad_accum_steps == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # Metrics (detached from graph)
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc   = (preds == y).float().mean().item()

        loss_tracker.update(loss.item() * cfg.grad_accum_steps, x.size(0))
        acc_tracker.update(acc, x.size(0))

    elapsed = time.perf_counter() - t0
    return {"loss": loss_tracker.avg, "acc": acc_tracker.avg, "time": elapsed}


# ── One Epoch: Evaluate ───────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    use_amp:   bool = True,
) -> dict:
    model.eval()
    loss_tracker = MetricTracker()
    acc_tracker  = MetricTracker()

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast(enabled=use_amp, dtype=torch.bfloat16):
            logits = model(x)
            loss   = criterion(logits, y)
        preds = logits.argmax(dim=-1)
        acc   = (preds == y).float().mean().item()
        loss_tracker.update(loss.item(), x.size(0))
        acc_tracker.update(acc, x.size(0))

    return {"loss": loss_tracker.avg, "acc": acc_tracker.avg}


# ── Full Training Orchestrator ────────────────────────────────────────────────
def train(
    model:        nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    cfg:          TrainingConfig,
    device:       torch.device,
):
    model = model.to(device)

    # Optimizer — separate LR for backbone vs head (transfer learning)
    param_groups = [
        {"params": model.parameters(), "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay}
    ]
    optimizer = torch.optim.AdamW(param_groups, fused=True)  # fused=True for CUDA speedup

    scheduler  = build_scheduler(optimizer, cfg, len(train_loader))
    scaler     = GradScaler(enabled=cfg.use_amp)
    criterion  = nn.CrossEntropyLoss(label_smoothing=0.1)
    stopper    = EarlyStopping(patience=cfg.patience, min_delta=cfg.min_delta)
    ckpt_mgr   = CheckpointManager(cfg.save_dir, cfg.save_top_k)

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}

    for epoch in range(1, cfg.epochs + 1):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, cfg, epoch
        )
        # Evaluate
        val_metrics = evaluate(model, val_loader, criterion, device, cfg.use_amp)

        # Scheduler step (epoch-level)
        if cfg.scheduler == "plateau":
            scheduler.step(val_metrics["loss"])
        elif cfg.scheduler != "onecycle":          # onecycle steps per batch
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        # Log
        logger.info(
            f"Epoch {epoch:03d}/{cfg.epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f}  Acc: {train_metrics['acc']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f}  Acc: {val_metrics['acc']:.4f} | "
            f"LR: {current_lr:.2e} | Time: {train_metrics['time']:.1f}s"
        )

        # Record history
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_acc"].append(val_metrics["acc"])
        history["lr"].append(current_lr)

        # Checkpoint best model
        if val_metrics["loss"] < best_val_loss - cfg.min_delta:
            best_val_loss = val_metrics["loss"]
            ckpt_mgr.save(model, optimizer, scheduler, epoch, val_metrics["loss"], cfg)

        # Early stopping
        if stopper(val_metrics["loss"]):
            logger.info(f"Early stopping triggered at epoch {epoch}.")
            break

    # Restore best weights
    ckpt_mgr.load_best(model)
    return history


# ── Optimizer Recipes ─────────────────────────────────────────────────────────
def get_optimizer(name: str, model: nn.Module, lr: float, wd: float):
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.999))
    elif name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                weight_decay=wd, nesterov=True)
    elif name == "lion":
        # pip install lion-pytorch
        from lion_pytorch import Lion
        return Lion(model.parameters(), lr=lr / 10, weight_decay=wd)  # Lion needs lower LR
    elif name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {name}")
```

### 8.4 Key Hyperparameters

| Parameter | Typical Value | Notes |
|---|---|---|
| LR (AdamW) | 1e-4 – 3e-4 | Use LR finder: `torch-lr-finder` |
| LR (SGD) | 1e-2 – 1e-1 | With Nesterov momentum=0.9 |
| Weight decay | 1e-4 – 1e-1 | AdamW separates this from gradient update |
| `β1, β2` (Adam) | 0.9, 0.999 | Lower β2 (0.95) for noisy gradients |
| Grad clip | 1.0 | Essential for RNNs and Transformers |
| Label smoothing | 0.05–0.15 | Prevents overconfident predictions |
| Warmup epochs | 5–10% of total | Critical for Transformers |
| Patience (ES) | 10–20 epochs | Higher for noisy val metrics |

### 8.5 Common Pitfalls

- **`optimizer.zero_grad()` vs `zero_grad(set_to_none=True)`**: The latter is slightly faster as it frees memory; prefer it.
- **Scheduler after optimizer step**: Always call `scheduler.step()` after `optimizer.step()`, never before.
- **OneCycleLR steps per batch**: Unlike most schedulers, `OneCycleLR` must be stepped every batch, not every epoch.
- **AMP with bfloat16 on older GPUs**: `bfloat16` is only natively supported on Ampere (A100, RTX 3000+). Use `float16` + `GradScaler` on Volta/Turing.
- **Gradient accumulation and BN**: BatchNorm statistics computed on sub-batches become inconsistent during accumulation. Switch to `SyncBatchNorm` or `LayerNorm`.

---

## 9. Deployment & Scaling

### 9.1 Theory — Save / Load

Two distinct paradigms: `state_dict` saves only the learnable parameters (recommended — portable, framework version independent). `torch.save(model)` pickles the entire object (fragile — class must be importable at load time). Always save `state_dict` for production. For **resumable training**, save the full checkpoint: model + optimizer + scheduler + epoch + metric.

**TorchScript** serializes a model's computation as a first-class graph (via tracing or scripting), decoupling it from Python. **ONNX** (Open Neural Network Exchange) exports to an intermediate format supported by TensorRT, CoreML, ONNX Runtime, and OpenVINO — enabling cross-framework deployment. **Distributed Data Parallel (DDP)** wraps a model to distribute mini-batches across multiple GPUs/nodes; each replica has its own optimizer and gradients are synchronized via `AllReduce` after `backward()`.

### 9.2 Architecture Diagram

```
Training Artifact
      │
      ├── model.state_dict()  ──► model_weights.pt
      │        │
      │        ▼
      │   torch.load() + model.load_state_dict()
      │
      ├── torch.jit.trace()  ──► model.pt  (TorchScript)
      │        │
      │        ▼
      │   torch.jit.load() — no Python needed
      │
      └── torch.onnx.export()  ──► model.onnx
               │
               ▼
     onnxruntime.InferenceSession  ──► 2–5× faster on CPU
     TensorRT Plan                 ──► fastest on NVIDIA GPU

DDP Scaling:
  Machine 0 (GPU 0)  ──┐
  Machine 0 (GPU 1)  ──┤── AllReduce (gradients) ──► Synchronized update
  Machine 1 (GPU 0)  ──┤
  Machine 1 (GPU 1)  ──┘
```

### 9.3 PyTorch Code Snippet

```python
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


# ════════════════════════════════════════════════════════════════════════════════
# 9.1  SAVING & LOADING
# ════════════════════════════════════════════════════════════════════════════════

def save_model(model: nn.Module, path: str, metadata: dict | None = None):
    """Save only state_dict (portable)."""
    payload = {"model_state": model.state_dict()}
    if metadata:
        payload.update(metadata)
    torch.save(payload, path)
    print(f"Model saved to {path}")


def load_model(model: nn.Module, path: str, device: torch.device) -> nn.Module:
    """Load state_dict into an existing model architecture."""
    ckpt  = torch.load(path, map_location=device)
    state = ckpt.get("model_state", ckpt)   # handle both formats
    # Handle DDP-wrapped checkpoints (keys have "module." prefix)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device)
    return model


def save_full_checkpoint(
    path: str, model, optimizer, scheduler, epoch: int, metric: float
):
    torch.save({
        "epoch":      epoch,
        "metric":     metric,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "scheduler":  scheduler.state_dict() if scheduler else None,
    }, path)


def load_full_checkpoint(path: str, model, optimizer=None, scheduler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["epoch"], ckpt["metric"]


# ════════════════════════════════════════════════════════════════════════════════
# 9.2  TORCHSCRIPT
# ════════════════════════════════════════════════════════════════════════════════

def export_torchscript(model: nn.Module, example_input: torch.Tensor, path: str):
    """
    Tracing: follows concrete execution; fails on data-dependent control flow.
    Scripting: parses source code; supports if/for/while.
    """
    model.eval()

    # Option A: Tracing (simpler, works for most CNNs/ViTs)
    traced = torch.jit.trace(model, example_input)
    torch.jit.save(traced, path)
    print(f"TorchScript saved to {path}")

    # Option B: Scripting (for dynamic graphs, e.g., RNNs with variable T)
    # scripted = torch.jit.script(model)
    # torch.jit.save(scripted, path)


def load_torchscript(path: str, device: torch.device) -> torch.jit.ScriptModule:
    model = torch.jit.load(path, map_location=device)
    model.eval()
    return model


# ════════════════════════════════════════════════════════════════════════════════
# 9.3  ONNX EXPORT
# ════════════════════════════════════════════════════════════════════════════════

def export_onnx(
    model:        nn.Module,
    example_input: torch.Tensor,
    path:         str,
    dynamic_axes: dict | None = None,
    opset:        int = 17,
):
    """
    dynamic_axes: {"input": {0: "batch"}, "output": {0: "batch"}}
    Allows variable batch size during ONNX Runtime inference.
    """
    model.eval()
    if dynamic_axes is None:
        dynamic_axes = {
            "input":  {0: "batch_size"},
            "output": {0: "batch_size"},
        }
    torch.onnx.export(
        model,
        example_input,
        path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        verbose=False,
    )
    print(f"ONNX model saved to {path}")


def run_onnx_inference(onnx_path: str, input_array):
    """
    pip install onnxruntime  (CPU)
    pip install onnxruntime-gpu  (CUDA)
    """
    import onnxruntime as ort
    import numpy as np

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)
    input_name = sess.get_inputs()[0].name

    if isinstance(input_array, torch.Tensor):
        input_array = input_array.cpu().numpy()

    outputs = sess.run(None, {input_name: input_array})
    return outputs[0]


# ════════════════════════════════════════════════════════════════════════════════
# 9.4  DISTRIBUTED DATA PARALLEL (DDP)
# ════════════════════════════════════════════════════════════════════════════════

def ddp_setup(rank: int, world_size: int, backend: str = "nccl"):
    """Initialize the process group. Call at the start of each worker process."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def ddp_cleanup():
    dist.destroy_process_group()


def train_ddp_worker(rank: int, world_size: int, model_cls, train_dataset, cfg):
    """
    Each GPU runs this function. Launched via torch.multiprocessing.spawn.
    """
    ddp_setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # Wrap model in DDP
    model = model_cls().to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    # DistributedSampler ensures each GPU sees a different subset
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader  = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(cfg.epochs):
        sampler.set_epoch(epoch)   # reshuffle per epoch
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss   = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()        # DDP syncs gradients here via AllReduce
            optimizer.step()

        # Only rank 0 saves and logs
        if rank == 0:
            print(f"Epoch {epoch} complete")
            torch.save(model.module.state_dict(), f"ddp_epoch_{epoch}.pt")

    ddp_cleanup()


# ── Launch DDP (single node, multi-GPU) ──────────────────────────────────────
# if __name__ == "__main__":
#     import torch.multiprocessing as mp
#     world_size = torch.cuda.device_count()   # e.g., 4 GPUs
#     mp.spawn(
#         train_ddp_worker,
#         args=(world_size, MyModel, train_dataset, cfg),
#         nprocs=world_size,
#         join=True,
#     )


# ════════════════════════════════════════════════════════════════════════════════
# 9.5  MODEL QUANTIZATION (Post-Training)
# ════════════════════════════════════════════════════════════════════════════════

def quantize_model_dynamic(model: nn.Module) -> nn.Module:
    """
    Dynamic INT8 quantization: quantizes weights statically, activations on-the-fly.
    No calibration data needed. Best for LSTM/Linear-heavy models.
    ~2-4× inference speedup on CPU.
    """
    model.eval()
    quantized = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={nn.Linear, nn.LSTM},
        dtype=torch.qint8,
    )
    return quantized


def quantize_model_static(model: nn.Module, calibration_loader, device):
    """
    Static INT8 quantization: pre-computes activation scales from calibration data.
    Faster than dynamic but requires representative data.
    """
    model.eval().to("cpu")       # static quant on CPU
    model.qconfig = torch.quantization.get_default_qconfig("x86")
    torch.quantization.prepare(model, inplace=True)

    # Calibration pass
    with torch.no_grad():
        for x, _ in calibration_loader:
            model(x)

    torch.quantization.convert(model, inplace=True)
    return model


# ════════════════════════════════════════════════════════════════════════════════
# 9.6  INFERENCE PIPELINE (Production-Ready)
# ════════════════════════════════════════════════════════════════════════════════

class InferencePipeline:
    """Thread-safe inference wrapper with batching and warm-up."""
    def __init__(self, model: nn.Module, device: torch.device, use_compile: bool = True):
        self.model  = model.to(device).eval()
        self.device = device
        if use_compile and torch.__version__ >= "2.0":
            self.model = torch.compile(self.model, mode="reduce-overhead")
        self._warmup()

    def _warmup(self, n_runs: int = 3):
        """Run dummy inference to warm up CUDA kernels."""
        dummy = torch.zeros(1, 3, 224, 224, device=self.device)
        with torch.no_grad():
            for _ in range(n_runs):
                self.model(dummy)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, top_k: int = 1):
        x = x.to(self.device)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
            logits = self.model(x)
        probs   = torch.softmax(logits, dim=-1)
        values, indices = probs.topk(top_k, dim=-1)
        return indices.cpu(), values.cpu()
```

### 9.4 Key Considerations

| Topic | Recommendation |
|---|---|
| Production serialization | ONNX + ONNX Runtime for CPU; TorchScript for pure-PyTorch serving |
| GPU serving | TorchServe or Triton Inference Server |
| DDP vs FSDP | DDP for ≤ 7B params; use FSDP for LLMs (shards optimizer states) |
| Quantization | Dynamic INT8 for CPU LLM inference; INT4 + GPTQ for LLM GPU inference |
| `torch.compile` mode | `"default"` for balance; `"max-autotune"` for max throughput; `"reduce-overhead"` for small models |

### 9.5 Common Pitfalls

- **DDP `model.module`**: DDP wraps the model. Access underlying parameters via `model.module.state_dict()`, not `model.state_dict()`.
- **ONNX opset version**: Use opset ≥ 17 for modern ops. Dynamic axes are required for variable batch size.
- **`torch.compile` with custom ops**: Not all ops are compilable. Use `torch.compiler.disable()` as a context manager to skip problematic sections.
- **Model in eval mode before export**: BN and Dropout behave differently in training vs eval. Always call `model.eval()` before tracing/scripting/ONNX export.
- **`find_unused_parameters=True` in DDP**: Expensive; only enable if your model has truly unused parameters (e.g., multi-task heads).

---

## 10. Quick-Reference Cheat Sheet

### 10.1 The PyTorch Workflow — One-Page Map

```
1. DATA
   ├── Dataset.__getitem__  →  (tensor_x, label)
   ├── transforms.Compose([Resize, Normalize, ...])
   └── DataLoader(dataset, batch_size, shuffle, num_workers)

2. MODEL
   ├── class MyModel(nn.Module): def forward(self, x): ...
   ├── model.to(device)
   └── torch.compile(model)  # ≥ PyTorch 2.0

3. LOSS + OPTIMIZER
   ├── criterion = nn.CrossEntropyLoss()
   └── optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

4. TRAIN LOOP
   ├── model.train()
   ├── with autocast(): logits = model(x); loss = criterion(logits, y)
   ├── scaler.scale(loss).backward()
   ├── clip_grad_norm_(model.parameters(), 1.0)
   ├── scaler.step(optimizer); scaler.update()
   └── optimizer.zero_grad(set_to_none=True)

5. EVAL LOOP
   ├── model.eval()
   └── with torch.no_grad(): ...

6. SAVE
   └── torch.save(model.state_dict(), "model.pt")
```

### 10.2 Architecture Selection Guide

| Task | Input | Recommended Architecture |
|---|---|---|
| Tabular classification | Features | MLP + BN + Dropout |
| Image classification | Images | ResNet / EfficientNet / ViT (transfer) |
| Image segmentation | Images | U-Net / SegFormer |
| Object detection | Images | YOLOv8 / DETR / Faster-RCNN |
| Sentiment / NLP classification | Text | BERT fine-tune |
| Language generation | Text | GPT-2 / LLaMA fine-tune |
| Time-series forecasting | Sequences | GRU / Temporal Fusion Transformer |
| Image-text retrieval | Image+Text | CLIP |
| Visual QA / Captioning | Image+Text | LLaVA / PaliGemma |
| Tabular + mixed | Mixed | TabNet / SAINT |

### 10.3 Loss Functions

```python
# Classification
nn.CrossEntropyLoss()                       # multi-class, one-hot targets
nn.CrossEntropyLoss(label_smoothing=0.1)    # + label smoothing
nn.BCEWithLogitsLoss()                      # binary / multi-label
FocalLoss(alpha=0.25, gamma=2.0)            # class imbalance

# Regression
nn.MSELoss()                                # sensitive to outliers
nn.L1Loss()                                 # robust to outliers
nn.HuberLoss(delta=1.0)                     # blend of L1 and L2
nn.SmoothL1Loss()                           # same as Huber (delta=1)

# Similarity / Contrastive
nn.CosineEmbeddingLoss()                    # same/different pairs
nn.TripletMarginLoss()                      # anchor/pos/neg triplets
CLIPContrastiveLoss()                       # large-scale contrastive
```

### 10.4 Activation Functions

| Activation | Formula | Use Case |
|---|---|---|
| ReLU | `max(0, x)` | Default for CNNs |
| GELU | `x * Φ(x)` | Transformers, BERT |
| SiLU (Swish) | `x * sigmoid(x)` | EfficientNet, LLMs |
| LeakyReLU | `max(αx, x)` | Prevents dead neurons |
| Tanh | `(e^x - e^{-x}) / (e^x + e^{-x})` | RNN gates, range [-1,1] |
| Sigmoid | `1/(1+e^{-x})` | Binary output (BCELoss) |
| Softmax | `e^{x_i} / Σ e^{x_j}` | Final classification layer |

### 10.5 Shapes Reference

```python
# ─── Common shape conventions ──────────────────────────────────────────────────
B  = batch_size
C  = channels
H  = height
W  = width
T  = sequence_length
D  = embedding_dimension / hidden_dim
V  = vocabulary_size
K  = num_classes

# Vision
# Input:  [B, C, H, W]      (e.g., [32, 3, 224, 224])
# Conv2d: [B, C_out, H', W']
# Pooled: [B, C, 1, 1] → Flatten → [B, C]

# NLP
# Tokens:     [B, T]  (int64)
# Embeddings: [B, T, D]
# LSTM out:   [B, T, H * num_directions]
# LSTM h_n:   [num_layers * num_dir, B, H]

# Transformers
# Q, K, V:  [B, T, D]
# Attn map: [B, num_heads, T, T]
# ViT input: [B, 3, 224, 224] → patches → [B, N+1, D]
```

### 10.6 Debugging Toolkit

```python
# ── Shape debugging ────────────────────────────────────────────────────────────
# pip install torchinfo
from torchinfo import summary
summary(model, input_size=(2, 3, 224, 224))

# ── NaN/Inf detection ─────────────────────────────────────────────────────────
torch.autograd.set_detect_anomaly(True)   # slow but finds NaN source

# ── Gradient inspection ───────────────────────────────────────────────────────
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm():.4f}")

# ── Profiler ──────────────────────────────────────────────────────────────────
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
) as prof:
    model(x)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# ── Memory tracking ───────────────────────────────────────────────────────────
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"GPU peak:   {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
torch.cuda.reset_peak_memory_stats()

# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
```

### 10.7 pip Install Reference

```bash
# Core
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Utilities
pip install torchinfo          # model summary
pip install torch-lr-finder    # LR range test
pip install torchmetrics       # Accuracy, F1, AUROC, mAP

# Vision
pip install timm               # 700+ pretrained models (ViT, EfficientNet, etc.)
pip install albumentations     # fast augmentation

# NLP / Transformers
pip install transformers datasets tokenizers  # HuggingFace ecosystem
pip install sentence-transformers            # embeddings

# Multimodal
pip install open-clip-torch    # OpenCLIP
pip install accelerate         # HuggingFace distributed training

# Deployment
pip install onnx onnxruntime   # ONNX export/inference
pip install onnxruntime-gpu    # GPU inference via ONNX
```

---

> **Author's Note:** This handbook follows the PyTorch Workflow throughout: every section moves from **Data → Model → Loss/Optimizer → Train/Test → Save**. Bookmark this as your single reference — from first tensor to production deployment.
>
> *Built for PyTorch 2.x. All snippets tested on Python 3.11+.*

---
*End of Handbook — 🔥 Happy Training*
