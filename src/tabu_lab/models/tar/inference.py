"""Independent test-row prediction. No cross-test-row evidence cache."""

from __future__ import annotations

import torch

from .episodes import episode_seed
from .types import TAREpisode


@torch.no_grad()
def predict_supervised(
    model,
    training_values,
    training_visible,
    test_covariates,
    test_visible,
    features,
    *,
    response_column=-1,
    codebook_seed=0,
    episode_ids=None,
):
    """Tables include a placeholder response column; hidden test response is erased.

    Each test row creates a separate episode with training facts + that row's
    visible covariates. Multiple test rows never become one another's evidence.
    codebook_seed is a root seed, with a distinct book for each episode ID.
    Pass stable episode_ids when reordering/chunking rows. Reusing IDs is replay.
    """
    if training_values.ndim != 2 or test_covariates.ndim != 2:
        raise ValueError("expected two matrices")
    m = training_values.shape[1]
    if test_covariates.shape[1] != m or not -m <= response_column < m:
        raise ValueError("response/schema width mismatch")
    a = response_column % m
    if (
        training_visible.shape != training_values.shape
        or test_visible.shape != test_covariates.shape
    ):
        raise ValueError("visible masks must match table shapes")
    ids = list(range(len(test_covariates))) if episode_ids is None else list(episode_ids)
    if len(ids) != len(test_covariates) or len(set(ids)) != len(ids):
        raise ValueError("episode IDs must be unique and match test rows")
    seeds = [episode_seed(codebook_seed, "supervised-inference", i) for i in ids]
    outputs = []
    was_training = model.training
    model.eval()
    try:
        for r in range(len(test_covariates)):
            values = torch.cat((training_values, test_covariates[r : r + 1]), dim=0)
            visible = torch.cat((training_visible, test_visible[r : r + 1]), dim=0).clone()
            visible[-1, a] = False
            queries = torch.zeros_like(visible)
            queries[-1, a] = True
            ep = TAREpisode.from_table(values, visible, queries, features, codebook_seed=seeds[r])
            outputs.append(model(ep).predictions[0])
    finally:
        model.train(was_training)
    return tuple(outputs)


@torch.no_grad()
def predict_joint_supervised(
    model,
    training_values,
    training_visible,
    test_covariates,
    test_visible,
    features,
    *,
    response_column=-1,
    codebook_seed=0,
    episode_id=0,
):
    """Predict the entire test set in one episode using the entire supplied train.

    All test labels are erased. Known test covariates are jointly visible, so
    predictions may depend on the other test covariates. This is a declared
    joint-test configuration, not equivalent to independent-row batching.
    """
    if training_values.ndim != 2 or test_covariates.ndim != 2:
        raise ValueError("expected two matrices")
    m = training_values.shape[1]
    if test_covariates.shape[1] != m or not -m <= response_column < m:
        raise ValueError("response/schema width mismatch")
    if (
        training_visible.shape != training_values.shape
        or test_visible.shape != test_covariates.shape
    ):
        raise ValueError("visible masks must match table shapes")
    a = response_column % m
    if not bool(training_visible[:, a].all()):
        raise ValueError("joint test prediction requires all training labels visible")
    if len(test_covariates) == 0:
        return ()
    values = torch.cat((training_values, test_covariates))
    visible = torch.cat((training_visible, test_visible)).clone()
    visible[len(training_values) :, a] = False
    queries = torch.zeros_like(visible)
    queries[len(training_values) :, a] = True
    episode = TAREpisode.from_table(
        values,
        visible,
        queries,
        features,
        codebook_seed=episode_seed(codebook_seed, "joint-supervised-inference", episode_id),
    )
    was_training = model.training
    model.eval()
    try:
        return model(episode).predictions
    finally:
        model.train(was_training)
