"""
Abstract Dataset
================
``BaseDataset`` is an abstract base class that standardises how any
tabular / flat-feature dataset is exposed to ``DeepLinearNetwork``.

Subclass it and implement the three abstract methods; everything else
(DataLoader creation, train/val splitting, feature-dimension reporting)
is provided for free.

Included concrete implementation
---------------------------------
``CsvDataset``  – loads any CSV where every column except the label is a
                  numeric feature.  Handles normalisation automatically.

Usage
-----
    # 1. Use the ready-made CSV loader
    from dataset import CsvDataset
    ds = CsvDataset("my_data.csv", label_column="target")
    train_loader, val_loader = ds.get_loaders(batch_size=64)

    # 2. Roll your own
    from dataset import BaseDataset
    class MyDataset(BaseDataset):
        def load(self): ...
        def __len__(self): ...
        def __getitem__(self, idx): ...
"""

from __future__ import annotations

import abc
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, random_split


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseDataset(Dataset, abc.ABC):
    """Abstract dataset compatible with ``DeepLinearNetwork``.

    Subclasses **must** implement:
        ``load()``        – called once in ``__init__`` to populate internal state.
        ``__len__()``     – number of samples.
        ``__getitem__()`` – returns ``(features_tensor, label_tensor)`` for one sample.

    Subclasses **may** override:
        ``feature_dim``   – property returning the number of input features.
        ``num_classes``   – property returning the number of target classes
                            (1 for regression / binary classification).
    """

    def __init__(self) -> None:
        super().__init__()
        self.load()

    @abc.abstractmethod
    def load(self) -> None:
        """Load / pre-process raw data into memory."""

    @abc.abstractmethod
    def __len__(self) -> int: ...

    @abc.abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]: ...

    # ------------------------------------------------------------------
    @property
    def feature_dim(self) -> int:
        """Number of input features (inferred from first sample by default)."""
        features, _ = self[0]
        return int(features.shape[0])

    @property
    def num_classes(self) -> int:
        """Number of output classes (inferred from first sample by default)."""
        _, label = self[0]
        return 1 if label.numel() == 1 else int(label.shape[0])

    # ------------------------------------------------------------------
    def get_loaders(
        self,
        batch_size: int = 32,
        val_split: float = 0.2,
        num_workers: int = 0,
        seed: int = 42,
    ) -> Tuple[DataLoader, DataLoader]:
        """Split dataset into train/val and return two DataLoaders.

        Args:
            batch_size:  Samples per batch.
            val_split:   Fraction of data to reserve for validation (0–1).
            num_workers: Worker processes for data loading (0 = main process).
            seed:        Random seed for reproducible splits.

        Returns:
            ``(train_loader, val_loader)``
        """
        val_size = int(len(self) * val_split)
        train_size = len(self) - val_size
        generator = torch.Generator().manual_seed(seed)
        train_ds, val_ds = random_split(self, [train_size, val_size],
                                         generator=generator)

        loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers)
        train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
        return train_loader, val_loader


# ---------------------------------------------------------------------------
# Ready-made CSV implementation
# ---------------------------------------------------------------------------

class CsvDataset(BaseDataset):
    """Load any numeric CSV file as a flat-feature dataset.

    The label column is separated from features; all remaining columns are
    cast to float32.  Features are z-score normalised (mean 0, std 1) using
    statistics computed on the full dataset.

    Args:
        csv_path:      Path to the CSV file.
        label_column:  Name of the target column.
        normalize:     If True (default), z-score normalize features.
    """

    def __init__(
        self,
        csv_path: str,
        label_column: str,
        normalize: bool = True,
    ) -> None:
        self.csv_path = csv_path
        self.label_column = label_column
        self.normalize = normalize
        super().__init__()          # calls self.load()

    # ------------------------------------------------------------------
    def load(self) -> None:
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("CsvDataset requires pandas: pip install pandas") from exc

        df = pd.read_csv(self.csv_path)

        labels = df[self.label_column].values
        features = df.drop(columns=[self.label_column]).select_dtypes("number").values

        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

        if self.normalize:
            mean = self.features.mean(dim=0)
            std = self.features.std(dim=0).clamp(min=1e-8)
            self.features = (self.features - mean) / std

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]
