import torch

from build_prototypes import fallback_display_name
from train import BalancedBatchSampler, stratified_split, supervised_contrastive_loss


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
