"""Create side-by-side UMAP projections for CE and SupCon checkpoints."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from torch.utils.data import DataLoader

from train import (
    FaceClassifier,
    FaceDataset,
    build_eval_transform,
    discover_images,
    embedding_geometry,
    load_face_classifier_state,
    stratified_split,
)


@torch.inference_mode()
def extract_embeddings(
    checkpoint: Path, paths: list[Path], labels: list[int], device: torch.device
) -> tuple[np.ndarray, torch.Tensor, dict[str, float]]:
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = FaceClassifier(
        len(state["class_names"]), int(state["config"]["embedding_dim"]), pretrained=None
    )
    load_face_classifier_state(model, state["model_state_dict"])
    model.to(device).eval()
    loader = DataLoader(
        FaceDataset(paths, labels, build_eval_transform()),
        batch_size=128,
        shuffle=False,
        num_workers=8,
        pin_memory=device.type == "cuda",
    )
    chunks = []
    for images, _ in loader:
        embeddings, _ = model(images.to(device, non_blocking=True))
        chunks.append(embeddings.cpu())
    embeddings = torch.cat(chunks)
    label_tensor = torch.tensor(labels)
    return embeddings.numpy(), label_tensor, embedding_geometry(embeddings, label_tensor)


def generate_umap_comparison(
    ce_checkpoint: Path,
    supcon_checkpoint: Path,
    data_dir: Path,
    output_path: Path,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> Path:
    paths, labels, _ = discover_images(data_dir)
    _, validation_indices = stratified_split(labels, validation_fraction, seed)
    validation_paths = [paths[index] for index in validation_indices]
    validation_labels = [labels[index] for index in validation_indices]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = (
        ("Cross-entropy", ce_checkpoint),
        ("Supervised contrastive", supcon_checkpoint),
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="#FFF8ED")
    for axis, (title, checkpoint) in zip(axes, models, strict=True):
        embeddings, label_tensor, metrics = extract_embeddings(
            checkpoint, validation_paths, validation_labels, device
        )
        projection = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            random_state=seed,
        ).fit_transform(embeddings)
        axis.set_facecolor("#FFF8ED")
        axis.scatter(
            projection[:, 0],
            projection[:, 1],
            c=label_tensor.numpy(),
            cmap="turbo",
            s=18,
            alpha=0.75,
            linewidths=0,
        )
        axis.set_title(
            f"{title}\n"
            f"intra ↓ {metrics['intra_class_distance']:.4f}  ·  "
            f"nearest centroid ↑ {metrics['nearest_centroid_distance']:.4f}  ·  "
            f"ratio ↑ {metrics['separation_ratio']:.2f}",
            color="#290D15",
            fontsize=15,
            fontweight="bold",
        )
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#D1962E")
            spine.set_linewidth(1.5)
    fig.suptitle(
        "Celebrity embedding geometry · identical validation samples",
        color="#290D15",
        fontsize=22,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.02,
        "Colors identify celebrity classes. UMAP is fitted independently per model "
        "with identical settings.",
        ha="center",
        color="#765D63",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path
