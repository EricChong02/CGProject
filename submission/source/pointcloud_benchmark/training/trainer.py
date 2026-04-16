"""Training loop for point cloud classification."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from pointcloud_benchmark.evaluation.metrics import compute_accuracy
from pointcloud_benchmark.utils.io import save_json


class Trainer:
    """Trainer for classification models in the benchmark pipeline."""

    def __init__(
        self,
        config: dict,
        model: torch.nn.Module,
        train_loader,
        val_loader,
        device: torch.device,
        logger,
    ) -> None:
        self.config = config
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.logger = logger

        training_cfg = config["training"]
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=training_cfg["learning_rate"],
            weight_decay=training_cfg["weight_decay"],
        )
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "best_val_acc": 0.0,
            "best_epoch": None,
        }

    def train(self) -> dict:
        epochs = self.config["training"]["epochs"]
        checkpoint_dir = Path(self.config["output"]["checkpoint_dir"])
        result_dir = Path(self.config["output"]["result_dir"])
        best_val_acc = float("-inf")

        for epoch in range(epochs):
            train_loss, train_acc = self._train_one_epoch(epoch)
            val_loss, val_acc = self._evaluate()
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            self.logger.info(
                "Epoch %d/%d | train_loss=%.4f | train_acc=%.4f | val_loss=%.4f | val_acc=%.4f",
                epoch + 1,
                epochs,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
            )

            latest_checkpoint_path = checkpoint_dir / "latest.pt"
            self._save_checkpoint(
                path=latest_checkpoint_path,
                epoch=epoch + 1,
                val_acc=val_acc,
                is_best=False,
            )

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                self.history["best_val_acc"] = val_acc
                self.history["best_epoch"] = epoch + 1
                best_checkpoint_path = checkpoint_dir / "best.pt"
                self._save_checkpoint(
                    path=best_checkpoint_path,
                    epoch=epoch + 1,
                    val_acc=val_acc,
                    is_best=True,
                )

        save_json(self.history, result_dir / "train_history.json")
        return self.history

    def _train_one_epoch(self, epoch: int) -> tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress = tqdm(self.train_loader, desc=f"Train {epoch + 1}", leave=False)
        for batch in progress:
            points, labels = self._prepare_batch(batch)

            logits = self.model(points)
            loss = self.criterion(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += batch_size

        average_loss = running_loss / max(total, 1)
        average_acc = correct / max(total, 1)
        return average_loss, average_acc

    @torch.no_grad()
    def _evaluate(self) -> tuple[float, float]:
        self.model.eval()
        predictions = []
        labels = []
        running_loss = 0.0
        total = 0
        for batch in self.val_loader:
            points, batch_labels = self._prepare_batch(batch)
            logits = self.model(points)
            loss = self.criterion(logits, batch_labels)
            predictions.append(logits.argmax(dim=1))
            labels.append(batch_labels)
            batch_size = batch_labels.size(0)
            running_loss += loss.item() * batch_size
            total += batch_size

        val_loss = running_loss / max(total, 1)
        val_acc = compute_accuracy(torch.cat(predictions), torch.cat(labels))
        return val_loss, val_acc

    def _prepare_batch(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        if "points" not in batch or "label" not in batch:
            raise KeyError("Each batch must contain 'points' and 'label' entries.")

        points = batch["points"]
        labels = batch["label"]
        if not isinstance(points, torch.Tensor) or not isinstance(labels, torch.Tensor):
            raise TypeError("Expected both batch['points'] and batch['label'] to be torch.Tensor objects.")
        if points.dim() != 3 or points.size(-1) != 3:
            raise ValueError(
                f"Expected point tensor shape [B, N, 3], got {tuple(points.shape)}."
            )
        if labels.dim() != 1:
            labels = labels.view(-1)
        if points.size(0) != labels.size(0):
            raise ValueError(
                f"Mismatched batch size between points and labels: {points.size(0)} vs {labels.size(0)}."
            )

        return points.to(self.device, dtype=torch.float32), labels.to(self.device, dtype=torch.long)

    def _save_checkpoint(self, path: Path, epoch: int, val_acc: float, is_best: bool) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_acc": val_acc,
            "model_name": self.config["model"]["name"],
            "dataset_name": self.config["dataset"]["name"],
            "config_path": self.config.get("config_path"),
            "history": self.history,
            "is_best": is_best,
        }
        torch.save(checkpoint, path)
