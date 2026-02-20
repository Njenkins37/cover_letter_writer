"""
Trainer
=======
A dataset-agnostic training loop for ``DeepLinearNetwork``.

Works with any ``BaseDataset`` subclass and any PyTorch loss function.
Automatically:
  • Uses GPU if available, otherwise CPU
  • Saves the best checkpoint (lowest val loss) to disk
  • Applies learning-rate scheduling (ReduceLROnPlateau)
  • Reports train / val loss and accuracy each epoch

Quick start
-----------
    from dataset import CsvDataset
    from neural_network import DeepLinearNetwork
    from train import Trainer

    ds = CsvDataset("data.csv", label_column="target")
    train_loader, val_loader = ds.get_loaders(batch_size=64)

    model = DeepLinearNetwork(input_size=ds.feature_dim, output_size=ds.num_classes)
    trainer = Trainer(model, task="classification")
    trainer.fit(train_loader, val_loader, epochs=50)
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    """Training loop for ``DeepLinearNetwork`` (or any ``nn.Module``).

    Parameters
    ----------
    model:
        The neural network to train.
    task:
        ``"classification"`` uses CrossEntropyLoss (multi-class) or
        BCEWithLogitsLoss (binary, output_size == 1).
        ``"regression"`` uses MSELoss.
    lr:
        Initial learning rate.
    weight_decay:
        L2 regularisation coefficient passed to Adam.
    checkpoint_path:
        Where to save the best model weights.  Pass ``None`` to skip saving.
    """

    def __init__(
        self,
        model: nn.Module,
        task: str = "classification",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        checkpoint_path: Optional[str] = "best_model.pt",
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.task = task
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )
        self.criterion = self._build_criterion()

        self._best_val_loss = float("inf")
        self._best_weights: Optional[dict] = None

        print(f"Trainer ready  |  device={self.device}  |  task={task}")

    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
    ) -> None:
        """Run the full training loop.

        Args:
            train_loader: DataLoader for training data.
            val_loader:   DataLoader for validation data.
            epochs:       Number of passes over the training set.
        """
        print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>10}  {'Val Acc':>8}  LR")
        print("-" * 55)

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_loss, val_acc = self._eval_epoch(val_loader)
            self.scheduler.step(val_loss)

            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                self._best_weights = copy.deepcopy(self.model.state_dict())
                if self.checkpoint_path:
                    torch.save(self._best_weights, self.checkpoint_path)
                marker = " *"
            else:
                marker = ""

            lr = self.optimizer.param_groups[0]["lr"]
            acc_str = f"{val_acc:.2%}" if val_acc is not None else "   N/A"
            print(
                f"{epoch:>6}  {train_loss:>10.4f}  {val_loss:>10.4f}"
                f"  {acc_str:>8}  {lr:.2e}{marker}"
            )

        print("\nTraining complete.")
        if self.checkpoint_path and self._best_weights:
            print(f"Best model saved to '{self.checkpoint_path}'  "
                  f"(val_loss={self._best_val_loss:.4f})")

    # ------------------------------------------------------------------
    def load_best(self) -> None:
        """Restore the best checkpoint weights into the model."""
        if self._best_weights:
            self.model.load_state_dict(self._best_weights)
        elif self.checkpoint_path and self.checkpoint_path.exists():
            self.model.load_state_dict(torch.load(self.checkpoint_path,
                                                   map_location=self.device))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for features, labels in loader:
            features, labels = features.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self._compute_loss(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item() * len(features)
        return total_loss / len(loader.dataset)

    def _eval_epoch(self, loader: DataLoader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in loader:
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = self.model(features)
                total_loss += self._compute_loss(outputs, labels).item() * len(features)
                if self.task == "classification":
                    preds = self._get_predictions(outputs)
                    target = labels.long() if labels.ndim == 1 else labels.argmax(dim=1)
                    correct += (preds == target).sum().item()
                    total += len(labels)
        avg_loss = total_loss / len(loader.dataset)
        accuracy = correct / total if self.task == "classification" else None
        return avg_loss, accuracy

    def _compute_loss(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.task == "regression":
            return self.criterion(outputs.squeeze(), labels.float())
        if outputs.shape[-1] == 1:          # binary classification
            return self.criterion(outputs.squeeze(), labels.float())
        return self.criterion(outputs, labels.long())

    def _get_predictions(self, outputs: torch.Tensor) -> torch.Tensor:
        if outputs.shape[-1] == 1:
            return (torch.sigmoid(outputs.squeeze()) >= 0.5).long()
        return outputs.argmax(dim=1)

    def _build_criterion(self) -> nn.Module:
        if self.task == "regression":
            return nn.MSELoss()
        return nn.CrossEntropyLoss()
