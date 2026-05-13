"""
PyTorch Models for Human Activity Recognition
Provides MLP and 1D-CNN architectures with training utilities.

MLP: Drop-in replacement for scikit-learn MLPClassifier with better training
     controls (batch training, LR scheduling, dropout).  Deploys via the
     existing NeuralNetworkCodeGenerator (weight extraction → C arrays).

CNN: 1D-CNN that operates on raw sensor windows (skips manual feature
     extraction).  Deployed via the dedicated CNNCodeGenerator which
     generates Conv1D + Pool + Dense layers in C / MicroPython.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy torch import — gracefully falls back when PyTorch is not installed
# ---------------------------------------------------------------------------
_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
except ImportError:
    logger.info("PyTorch not installed — pytorch_mlp / pytorch_cnn model "
                "types will be unavailable.")
    # Provide stubs so the class definitions parse without errors
    class _ModuleStub:
        """Placeholder base class when torch is absent."""
        pass

    class nn:  # noqa: N801
        Module = _ModuleStub
        Linear = None
        Conv1d = None
        MaxPool1d = None
        Sequential = None
        ReLU = None
        Dropout = None

    torch = None  # type: ignore
    optim = None  # type: ignore


def is_pytorch_available() -> bool:
    """Check whether PyTorch is importable."""
    return _TORCH_AVAILABLE


# ===================================================================
# Model definitions
# ===================================================================

class HARMLP(nn.Module):
    """Multi-Layer Perceptron for HAR (feature-based input).

    Architecture mirrors scikit-learn MLPClassifier:
      Input → [Hidden_i → ReLU → Dropout]* → Output (logits)
    """

    def __init__(self, input_size: int, hidden_sizes: Tuple[int, ...],
                 num_classes: int, dropout: float = 0.3):
        super().__init__()
        layers: list = []
        prev = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class HARCNN(nn.Module):
    """1D-CNN for HAR operating directly on raw sensor windows.

    Input shape: (batch, window_size, n_channels)  — typically n_channels=6
    Architecture:
      Conv1D(32, k=5) → ReLU → Conv1D(64, k=5) → ReLU → MaxPool(2)
      → Conv1D(128, k=3) → ReLU → GlobalAvgPool → Dense → Output
    """

    def __init__(self, window_size: int, n_channels: int, num_classes: int,
                 dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, 32,  kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64,  kernel_size=5, padding=2)
        self.pool  = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.drop  = nn.Dropout(dropout)

        # After pool(2) the time dimension halves once: window_size // 2
        # Global average pooling collapses time → 128 features
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, window_size, channels) → need (batch, channels, window_size)
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.drop(x)
        # Global average pooling over time dimension
        x = x.mean(dim=2)          # (batch, 128)
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x


class HARCNN2D(nn.Module):
    """2D-CNN for HAR treating the sensor window as a time × channel image.

    Input shape: (batch, window_size, n_channels)
    Internally reshaped to (batch, 1, window_size, n_channels) so Conv2D
    kernels learn both temporal patterns and cross-channel correlations.

    Architecture (3-stage):
      Stage 1 — temporal feature extraction (per-channel):
        Conv2D(1 → 32,  k=5×1, pad=(2,0)) → BN → ReLU
        Conv2D(32 → 64, k=5×1, pad=(2,0)) → BN → ReLU
        MaxPool2D(2×1)   — halves time, keeps channel dim

      Stage 2 — cross-channel fusion:
        Conv2D(64 → 128, k=3×n_channels, pad=(1,0)) → BN → ReLU
        (collapses channel dim → output shape: batch × 128 × T' × 1)

      Stage 3 — classifier:
        AdaptiveAvgPool2D(1×1) → Flatten(128) → Dropout → Dense(64) → ReLU → Dropout → Output

    Advantages over 1D-CNN:
      • Captures correlations between sensor axes (e.g. aX↔gX coupling)
      • Stage-1 kernels are parameter-efficient: same as 1D per-axis Conv
      • Stage-2 performs explicit sensor-fusion in one learnable operation
    """

    def __init__(self, window_size: int, n_channels: int, num_classes: int,
                 dropout: float = 0.3):
        super().__init__()
        self.n_channels = n_channels

        # Stage 1: temporal conv per-channel (kernel spans time, NOT channels)
        self.conv1 = nn.Conv2d(1,  32, kernel_size=(5, 1), padding=(2, 0))
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 1), padding=(2, 0))
        self.bn2   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(kernel_size=(2, 1))   # halve time only

        # Stage 2: cross-channel fusion (kernel spans ALL channels)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, n_channels), padding=(1, 0))
        self.bn3   = nn.BatchNorm2d(128)

        # Stage 3: classifier
        self.gap  = nn.AdaptiveAvgPool2d((1, 1))        # global avg → (batch, 128, 1, 1)
        self.drop = nn.Dropout(dropout)
        self.fc1  = nn.Linear(128, 64)
        self.fc2  = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, window_size, n_channels)
        x = x.unsqueeze(1)                  # → (batch, 1, window_size, n_channels)

        x = self.relu(self.bn1(self.conv1(x)))  # (batch, 32, T, C)
        x = self.relu(self.bn2(self.conv2(x)))  # (batch, 64, T, C)
        x = self.pool(x)                        # (batch, 64, T/2, C)

        x = self.relu(self.bn3(self.conv3(x)))  # (batch, 128, T/2, 1)

        x = self.gap(x).flatten(1)              # (batch, 128)
        x = self.drop(x)
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x


# ===================================================================
# Trainer
# ===================================================================

class PyTorchTrainer:
    """Unified training loop for HARMLP and HARCNN.

    Handles batching, LR scheduling, early stopping, and weight export
    in a format compatible with the existing code generators.
    """

    def __init__(self, model: nn.Module, device: Optional[str] = None):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required but not installed. "
                               "Install via: pip install torch")
        self.model = model
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.train_losses: List[float] = []
        self.val_accuracies: List[float] = []
        self.train_accuracies: List[float] = []

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              epochs: int = 200, batch_size: int = 32,
              lr: float = 1e-3, patience: int = 15,
              weight_decay: float = 1e-4) -> Dict[str, Any]:
        """Train the model with mini-batch SGD + cosine-annealing LR schedule.

        Returns a metrics dict compatible with EdgeMLModel.performance_metrics.
        """
        self.model.train()

        # Build DataLoaders
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.long)
        train_ds = TensorDataset(X_t, y_t)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        optimizer = optim.AdamW(self.model.parameters(), lr=lr,
                                weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0.0
        best_state = None
        patience_counter = 0

        for epoch in range(epochs):
            # --- mini-batch training step ---
            running_loss = 0.0
            self.model.train()
            for xb, yb in train_dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * xb.size(0)

            scheduler.step()
            avg_loss = running_loss / len(train_ds)
            self.train_losses.append(avg_loss)

            # --- train accuracy ---
            train_acc = self._accuracy(X_train, y_train)
            self.train_accuracies.append(train_acc)

            # --- validation early stopping ---
            if X_val is not None and y_val is not None:
                val_acc = self._accuracy(X_val, y_val)
                self.val_accuracies.append(val_acc)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = {k: v.cpu().clone()
                                  for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                    logger.info(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}  "
                                f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f} (best)")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                # No validation — just track training
                if epoch % 20 == 0 or epoch == epochs - 1:
                    logger.info(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}  "
                                f"train_acc={train_acc:.4f}")

        # Restore best model (if validation was used)
        if best_state is not None:
            self.model.load_state_dict(best_state)

        metrics: Dict[str, Any] = {
            "train_accuracy": self.train_accuracies[-1],
            "train_losses": self.train_losses,
            "train_accuracies": self.train_accuracies,
        }
        if X_val is not None:
            metrics["val_accuracies"] = self.val_accuracies
            metrics["best_val_accuracy"] = best_val_acc
            metrics["early_stopped"] = patience_counter >= patience
            metrics["stopped_epoch"] = epoch + 1
        return metrics

    # ------------------------------------------------------------------ #
    # Inference helpers
    # ------------------------------------------------------------------ #

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class indices."""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(
                torch.tensor(X, dtype=torch.float32).to(self.device))
            return logits.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities (softmax)."""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(
                torch.tensor(X, dtype=torch.float32).to(self.device))
            return torch.softmax(logits, dim=1).cpu().numpy()

    def _accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return float(np.mean(preds == y))

    # ------------------------------------------------------------------ #
    # Weight export (for code generators)
    # ------------------------------------------------------------------ #

    def export_mlp_weights(self) -> Dict[str, Any]:
        """Export MLP weights in scikit-learn–compatible format.

        Returns dict with lists of weight matrices (coefs_) and bias vectors
        (intercepts_) plus hidden_layer_sizes — identical structure to what
        NeuralNetworkCodeGenerator._extract_real_weights() consumes from
        an sklearn MLPClassifier.
        """
        self.model.eval()
        coefs: List[np.ndarray] = []
        intercepts: List[np.ndarray] = []
        hidden_sizes: List[int] = []

        for module in self.model.net:
            if isinstance(module, nn.Linear):
                # nn.Linear stores weight as (out_features, in_features)
                # scikit-learn coefs_ are (in_features, out_features)
                w = module.weight.detach().cpu().numpy().T
                b = module.bias.detach().cpu().numpy()
                coefs.append(w)
                intercepts.append(b)

        # hidden_layer_sizes = sizes of all hidden layers (exclude output)
        for i, (c, _) in enumerate(zip(coefs, intercepts)):
            if i < len(coefs) - 1:  # not the output layer
                hidden_sizes.append(c.shape[1])

        return {
            "coefs_": coefs,
            "intercepts_": intercepts,
            "hidden_layer_sizes": tuple(hidden_sizes),
        }

    def export_cnn_weights(self) -> Dict[str, Any]:
        """Export CNN weights for CNNCodeGenerator.

        Returns a structured dict describing every layer.  Handles both
        Conv1d (HARCNN) and Conv2d (HARCNN2D) layers.
        """
        self.model.eval()
        layers: List[Dict[str, Any]] = []

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv1d):
                w = module.weight.detach().cpu().numpy()  # (out_ch, in_ch, k)
                b = module.bias.detach().cpu().numpy() if module.bias is not None else np.zeros(w.shape[0])
                layers.append({
                    "type": "conv1d",
                    "name": name,
                    "weights": w.tolist(),
                    "bias": b.tolist(),
                    "out_channels": w.shape[0],
                    "in_channels": w.shape[1],
                    "kernel_size": w.shape[2],
                    "padding": module.padding[0],
                })
            elif isinstance(module, nn.Conv2d):
                w = module.weight.detach().cpu().numpy()  # (out_ch, in_ch, kH, kW)
                b = module.bias.detach().cpu().numpy() if module.bias is not None else np.zeros(w.shape[0])
                pad = module.padding
                layers.append({
                    "type": "conv2d",
                    "name": name,
                    "weights": w.tolist(),
                    "bias": b.tolist(),
                    "out_channels": w.shape[0],
                    "in_channels": w.shape[1],
                    "kernel_h": w.shape[2],
                    "kernel_w": w.shape[3],
                    "padding_h": pad[0] if isinstance(pad, tuple) else pad,
                    "padding_w": pad[1] if isinstance(pad, tuple) else pad,
                })
            elif isinstance(module, nn.MaxPool1d):
                layers.append({
                    "type": "maxpool1d",
                    "name": name,
                    "kernel_size": module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0],
                })
            elif isinstance(module, nn.MaxPool2d):
                ks = module.kernel_size
                layers.append({
                    "type": "maxpool2d",
                    "name": name,
                    "kernel_h": ks[0] if isinstance(ks, tuple) else ks,
                    "kernel_w": ks[1] if isinstance(ks, tuple) else ks,
                })
            elif isinstance(module, nn.AdaptiveAvgPool2d):
                layers.append({
                    "type": "adaptive_avgpool2d",
                    "name": name,
                    "output_size": list(module.output_size),
                })
            elif isinstance(module, nn.BatchNorm2d):
                layers.append({
                    "type": "batchnorm2d",
                    "name": name,
                    "num_features": module.num_features,
                    "weight": module.weight.detach().cpu().numpy().tolist(),
                    "bias": module.bias.detach().cpu().numpy().tolist(),
                    "running_mean": module.running_mean.detach().cpu().numpy().tolist(),
                    "running_var": module.running_var.detach().cpu().numpy().tolist(),
                    "eps": module.eps,
                })
            elif isinstance(module, nn.Linear):
                w = module.weight.detach().cpu().numpy()  # (out, in)
                b = module.bias.detach().cpu().numpy() if module.bias is not None else np.zeros(w.shape[0])
                layers.append({
                    "type": "dense",
                    "name": name,
                    "weights": w.T.tolist(),   # store as (in, out) like sklearn
                    "bias": b.tolist(),
                    "in_features": w.shape[1],
                    "out_features": w.shape[0],
                })

        return {"layers": layers}
