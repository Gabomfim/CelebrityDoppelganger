"""Fine-tune a face-recognition backbone with supervised contrastive learning."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
from PIL import Image, UnidentifiedImageError
from torch import Tensor, nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class TrainConfig:
    data_dir: str = "data/source_dataset"
    output_dir: str = "models"
    epochs: int = 30
    batch_size: int = 64
    classes_per_batch: int = 16
    samples_per_class: int = 4
    learning_rate: float = 3e-4
    backbone_learning_rate: float = 3e-5
    weight_decay: float = 1e-4
    temperature: float = 0.07
    contrastive_weight: float = 0.5
    validation_fraction: float = 0.2
    embedding_dim: int = 256
    num_workers: int = 4
    seed: int = 42
    patience: int = 7
    wandb_project: str = "celebrity-doppelganger"
    wandb_entity: str | None = None
    run_name: str | None = None
    resume: str | None = None
    offline: bool = False
    full_data: bool = False


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def discover_images(root: Path) -> tuple[list[Path], list[int], list[str]]:
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    class_names = sorted(path.name for path in root.iterdir() if path.is_dir())
    paths: list[Path] = []
    labels: list[int] = []
    kept_names: list[str] = []
    for class_name in class_names:
        images = sorted(
            path
            for path in (root / class_name).iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not images:
            continue
        label = len(kept_names)
        kept_names.append(class_name)
        paths.extend(images)
        labels.extend([label] * len(images))
    if len(kept_names) < 2:
        raise ValueError("At least two non-empty identity directories are required")
    return paths, labels, kept_names


def stratified_split(
    labels: list[int], validation_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    by_class: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_class[label].append(index)
    rng = random.Random(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for indices in by_class.values():
        rng.shuffle(indices)
        validation_count = (
            min(len(indices) - 1, max(1, round(len(indices) * validation_fraction)))
            if len(indices) > 1
            else 0
        )
        validation_indices.extend(indices[:validation_count])
        train_indices.extend(indices[validation_count:])
    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return train_indices, validation_indices


def build_train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                160, scale=(0.75, 1.0), ratio=(0.9, 1.1), interpolation=InterpolationMode.BILINEAR
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.15, 0.05)], p=0.7),
            transforms.RandomGrayscale(p=0.05),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.5))], p=0.15),
            transforms.RandomRotation(8, interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.12), ratio=(0.5, 2.0)),
        ]
    )


def build_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(176, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(160),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class FaceDataset(Dataset[tuple[Tensor, int]]):
    def __init__(self, paths: list[Path], labels: list[int], transform: transforms.Compose):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        try:
            with Image.open(self.paths[index]) as image:
                tensor = self.transform(image.convert("RGB"))
        except (OSError, UnidentifiedImageError) as error:
            raise RuntimeError(f"Could not load image {self.paths[index]}") from error
        return tensor, self.labels[index]


class BalancedBatchSampler(Sampler[list[int]]):
    """Yield P identities x K images, sampling rare identities with replacement."""

    def __init__(
        self, labels: list[int], classes_per_batch: int, samples_per_class: int, seed: int
    ) -> None:
        self.by_class: dict[int, list[int]] = defaultdict(list)
        for index, label in enumerate(labels):
            self.by_class[label].append(index)
        self.classes = sorted(self.by_class)
        self.classes_per_batch = min(classes_per_batch, len(self.classes))
        self.samples_per_class = samples_per_class
        self.batch_size = self.classes_per_batch * samples_per_class
        self.num_batches = max(1, math.ceil(len(labels) / self.batch_size))
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return self.num_batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        for _ in range(self.num_batches):
            selected_classes = rng.sample(self.classes, self.classes_per_batch)
            batch = []
            for label in selected_classes:
                choices = self.by_class[label]
                if len(choices) >= self.samples_per_class:
                    batch.extend(rng.sample(choices, self.samples_per_class))
                else:
                    batch.extend(rng.choices(choices, k=self.samples_per_class))
            rng.shuffle(batch)
            yield batch


class FaceClassifier(nn.Module):
    def __init__(
        self, num_classes: int, embedding_dim: int = 256, pretrained: str | None = "vggface2"
    ) -> None:
        super().__init__()
        self.backbone = InceptionResnetV1(pretrained=pretrained, classify=False)
        self.projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        features = self.backbone(images)
        embeddings = F.normalize(self.projector(features), dim=1)
        return embeddings, self.classifier(embeddings)


def load_face_classifier_state(model: nn.Module, state_dict: Mapping[str, Tensor]) -> None:
    """Load a checkpoint while ignoring unused pretrained FaceNet logits."""
    model_keys = set(model.state_dict())
    unexpected = set(state_dict) - model_keys
    allowed_unused = {"backbone.logits.weight", "backbone.logits.bias"}
    if unexpected - allowed_unused:
        names = ", ".join(sorted(unexpected - allowed_unused))
        raise RuntimeError(f"Unexpected checkpoint keys: {names}")
    model.load_state_dict({key: value for key, value in state_dict.items() if key in model_keys})


def supervised_contrastive_loss(features: Tensor, labels: Tensor, temperature: float) -> Tensor:
    """Supervised contrastive loss from Khosla et al., excluding self-pairs."""
    features = F.normalize(features, dim=1)
    logits = features @ features.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    identity = torch.eye(labels.shape[0], dtype=torch.bool, device=labels.device)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~identity
    logits_mask = ~identity
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    if not valid.any():
        return features.sum() * 0.0
    mean_log_prob = (positive_mask * log_prob).sum(dim=1) / positive_count.clamp_min(1)
    return -mean_log_prob[valid].mean()


def make_optimizer(model: FaceClassifier, config: TrainConfig) -> torch.optim.Optimizer:
    head_parameters = list(model.projector.parameters()) + list(model.classifier.parameters())
    return torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": config.backbone_learning_rate},
            {"params": head_parameters, "lr": config.learning_rate},
        ],
        weight_decay=config.weight_decay,
    )


def run_epoch(
    model: FaceClassifier,
    loader: DataLoader,
    device: torch.device,
    config: TrainConfig,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: GradScaler | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = defaultdict(float)
    sample_count = 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with (
            torch.set_grad_enabled(training),
            torch.autocast(device_type=device.type, enabled=device.type == "cuda"),
        ):
            embeddings, logits = model(images)
            ce_loss = F.cross_entropy(logits, labels)
            contrastive_loss = supervised_contrastive_loss(embeddings, labels, config.temperature)
            loss = ce_loss + config.contrastive_weight * contrastive_loss
        if training and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        count = labels.shape[0]
        sample_count += count
        totals["loss"] += loss.item() * count
        totals["ce_loss"] += ce_loss.item() * count
        totals["contrastive_loss"] += contrastive_loss.item() * count
        totals["accuracy"] += (logits.argmax(dim=1) == labels).sum().item()
    return {name: value / max(1, sample_count) for name, value in totals.items()}


def checkpoint_payload(
    model: FaceClassifier,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_accuracy: float,
    class_names: list[str],
    config: TrainConfig,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "epoch": epoch,
        "best_validation_accuracy": best_accuracy,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "class_names": class_names,
        "config": asdict(config),
    }


def train(config: TrainConfig) -> Path:
    import wandb

    seed_everything(config.seed)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    paths, labels, class_names = discover_images(Path(config.data_dir))
    if config.full_data:
        train_indices, validation_indices = list(range(len(paths))), []
    else:
        train_indices, validation_indices = stratified_split(
            labels, config.validation_fraction, config.seed
        )
    train_paths = [paths[index] for index in train_indices]
    train_labels = [labels[index] for index in train_indices]
    validation_paths = [paths[index] for index in validation_indices]
    validation_labels = [labels[index] for index in validation_indices]
    sampler = BalancedBatchSampler(
        train_labels, config.classes_per_batch, config.samples_per_class, config.seed
    )
    effective_batch_size = sampler.batch_size
    if config.batch_size != effective_batch_size:
        print(f"Using balanced batch size {effective_batch_size} (P x K sampler)")
    loader_kwargs = {
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": config.num_workers > 0,
    }
    train_loader = DataLoader(
        FaceDataset(train_paths, train_labels, build_train_transform()),
        batch_sampler=sampler,
        **loader_kwargs,
    )
    validation_loader = (
        None
        if config.full_data
        else DataLoader(
            FaceDataset(validation_paths, validation_labels, build_eval_transform()),
            batch_size=effective_batch_size,
            shuffle=False,
            **loader_kwargs,
        )
    )
    model = FaceClassifier(len(class_names), config.embedding_dim).to(device)
    optimizer = make_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = GradScaler(enabled=device.type == "cuda")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "classes.json").write_text(
        json.dumps(class_names, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    start_epoch, best_accuracy, stale_epochs = 0, -1.0, 0
    if config.resume:
        state = torch.load(config.resume, map_location=device, weights_only=False)
        load_face_classifier_state(model, state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        start_epoch = int(state["epoch"]) + 1
        best_accuracy = float(state["best_validation_accuracy"])
    run = wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.run_name,
        config=asdict(config),
        mode="offline" if config.offline else os.getenv("WANDB_MODE", "online"),
        tags=[
            "supervised-contrastive",
            "facenet",
            "vggface2",
            "full-data-refit" if config.full_data else "evaluation-split",
        ],
    )
    run.config.update(
        {
            "num_classes": len(class_names),
            "train_images": len(train_paths),
            "validation_images": len(validation_paths),
            "device": str(device),
        }
    )
    best_path = output_dir / "best.pt"
    for epoch in range(start_epoch, config.epochs):
        sampler.set_epoch(epoch)
        train_metrics = run_epoch(model, train_loader, device, config, optimizer, scaler)
        validation_metrics = (
            None
            if validation_loader is None
            else run_epoch(model, validation_loader, device, config)
        )
        scheduler.step()
        metrics = {
            **{f"train/{key}": value for key, value in train_metrics.items()},
            "epoch": epoch,
            "learning_rate/backbone": optimizer.param_groups[0]["lr"],
            "learning_rate/head": optimizer.param_groups[1]["lr"],
        }
        if validation_metrics is not None:
            metrics.update(
                {f"validation/{key}": value for key, value in validation_metrics.items()}
            )
        run.log(metrics, step=epoch)
        payload = checkpoint_payload(
            model, optimizer, scheduler, epoch, best_accuracy, class_names, config
        )
        epoch_path = output_dir / f"checkpoint-{epoch + 1:03d}.pt"
        torch.save(payload, epoch_path)
        current_accuracy = validation_metrics["accuracy"] if validation_metrics else None
        if config.full_data:
            torch.save(payload, best_path)
        elif current_accuracy is not None and current_accuracy > best_accuracy:
            best_accuracy, stale_epochs = current_accuracy, 0
            payload["best_validation_accuracy"] = best_accuracy
            torch.save(payload, best_path)
        else:
            stale_epochs += 1
        validation_summary = (
            f" val_loss={validation_metrics['loss']:.4f} val_acc={current_accuracy:.4f}"
            if validation_metrics is not None and current_accuracy is not None
            else " full_data=true"
        )
        print(
            f"epoch={epoch + 1}/{config.epochs} train_loss={train_metrics['loss']:.4f}"
            f"{validation_summary}"
        )
        if not config.full_data and stale_epochs >= config.patience:
            print(f"Early stopping after {config.patience} epochs without improvement")
            break
    best_state = torch.load(best_path, map_location="cpu", weights_only=False)
    final_path = output_dir / "celebrity_face_classifier.pt"
    torch.save(best_state, final_path)
    artifact = wandb.Artifact(
        "celebrity-face-classifier",
        type="model",
        metadata={
            "format_version": 1,
            "backbone": "InceptionResnetV1-vggface2",
            "best_validation_accuracy": (
                None if config.full_data else best_state["best_validation_accuracy"]
            ),
            "training_scope": "full_dataset" if config.full_data else "evaluation_split",
            "num_classes": len(class_names),
        },
    )
    artifact.add_file(str(final_path))
    artifact.add_file(str(output_dir / "classes.json"))
    run.log_artifact(artifact, aliases=["latest", "best"])
    if not config.full_data:
        run.summary["best_validation_accuracy"] = best_state["best_validation_accuracy"]
    run.summary["training_scope"] = "full_dataset" if config.full_data else "evaluation_split"
    run.finish()
    return final_path


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    for field_name, field in TrainConfig.__dataclass_fields__.items():
        default = field.default
        argument = f"--{field_name.replace('_', '-')}"
        if isinstance(default, bool):
            parser.add_argument(argument, action="store_true", default=default)
        else:
            value_type = type(default) if default is not None else str
            parser.add_argument(argument, type=value_type, default=default)
    return TrainConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    train(parse_args())
