"""Training loop placeholder."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from pointcloud_benchmark.evaluation.metrics import compute_accuracy
from pointcloud_benchmark.utils.io import save_json


class Trainer:
    """Minimal trainer for smoke-testing the pipeline."""

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
        self.history = {"train_loss": [], "train_acc": [], "val_acc": []}

    def train(self) -> dict:
        epochs = self.config["training"]["epochs"]
        checkpoint_dir = Path(self.config["output"]["checkpoint_dir"])
        result_dir = Path(self.config["output"]["result_dir"])

        for epoch in range(epochs):
            train_loss, train_acc = self._train_one_epoch(epoch)
            val_acc = self._evaluate()
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            self.logger.info(
                "Epoch %d/%d | train_loss=%.4f | train_acc=%.4f | val_acc=%.4f",
                epoch + 1,
                epochs,
                train_loss,
                train_acc,
                val_acc,
            )

        checkpoint_path = checkpoint_dir / "latest.pt"
        torch.save({"model_state_dict": self.model.state_dict()}, checkpoint_path)
        save_json(self.history, result_dir / "train_history.json")
        return self.history

    def _train_one_epoch(self, epoch: int) -> tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress = tqdm(self.train_loader, desc=f"Train {epoch + 1}", leave=False)
        for batch in progress:
            points = batch["points"].to(self.device)
            labels = batch["label"].to(self.device)

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
    def _evaluate(self) -> float:
        self.model.eval()
        predictions = []
        labels = []
        for batch in self.val_loader:
            points = batch["points"].to(self.device)
            batch_labels = batch["label"].to(self.device)
            logits = self.model(points)
            predictions.append(logits.argmax(dim=1))
            labels.append(batch_labels)
        return compute_accuracy(torch.cat(predictions), torch.cat(labels))

