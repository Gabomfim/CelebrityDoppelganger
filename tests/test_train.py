import pytest
import torch
from torch import nn

from build_prototypes import fallback_display_name
from train import (
    FACENET_MEAN,
    FACENET_STD,
    BalancedBatchSampler,
    embedding_geometry,
    load_face_classifier_state,
    stratified_split,
    supervised_contrastive_loss,
)


def test_facenet_preprocessing_matches_fixed_image_standardization():
    assert pytest.approx((0.5, 0.5, 0.5)) == FACENET_MEAN
    assert pytest.approx((128 / 255,) * 3) == FACENET_STD


def test_embedding_geometry_rewards_compact_separated_clusters():
    labels = torch.tensor([0, 0, 1, 1])
    compact = torch.tensor([[1.0, 0.0], [0.99, 0.01], [-1.0, 0.0], [-0.99, -0.01]])
    overlapping = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2], [0.2, 0.8]])

    compact_metrics = embedding_geometry(compact, labels)
    overlapping_metrics = embedding_geometry(overlapping, labels)

    assert compact_metrics["intra_class_distance"] < overlapping_metrics["intra_class_distance"]
    assert (
        compact_metrics["nearest_centroid_distance"]
        > overlapping_metrics["nearest_centroid_distance"]
    )
    assert compact_metrics["separation_ratio"] > overlapping_metrics["separation_ratio"]


def test_supervised_contrastive_loss_is_finite_and_differentiable():
    features = torch.randn(8, 16, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss = supervised_contrastive_loss(features, labels, temperature=0.07)
    loss.backward()
    assert torch.isfinite(loss)
    assert features.grad is not None


def test_balanced_sampler_has_repeated_examples_per_class():
    labels = [0, 1, 1, 2, 2, 2]
    sampler = BalancedBatchSampler(labels, classes_per_batch=3, samples_per_class=2, seed=1)
    batch = next(iter(sampler))
    batch_labels = [labels[index] for index in batch]
    assert len(batch) == 6
    assert all(batch_labels.count(label) == 2 for label in set(batch_labels))


def test_stratified_split_keeps_singletons_in_training():
    train_indices, validation_indices = stratified_split([0, 1, 1, 1], 0.2, seed=1)
    assert 0 in train_indices
    assert 0 not in validation_indices


def test_fallback_display_name_removes_underscores():
    assert fallback_display_name("samuel_l_jackson") == "Samuel L Jackson"


def test_checkpoint_loader_ignores_only_unused_facenet_logits():
    model = nn.Linear(2, 1)
    checkpoint = {
        **model.state_dict(),
        "backbone.logits.weight": torch.randn(2, 2),
        "backbone.logits.bias": torch.randn(2),
    }
    load_face_classifier_state(model, checkpoint)
    with pytest.raises(RuntimeError, match="Unexpected checkpoint keys"):
        load_face_classifier_state(model, {**model.state_dict(), "unexpected": torch.ones(1)})
