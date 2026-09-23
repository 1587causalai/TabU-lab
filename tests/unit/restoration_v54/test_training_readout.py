"""Loss-address pruning must preserve admission, gradients, and optimizer state."""

import copy

import pytest
import torch

from tabu_lab.models.restoration.contracts import TruthSidecar, make_episode
from tabu_lab.models.restoration.end_to_end_checks import example_episode
from tabu_lab.models.restoration_v53.training import prepare_training_episode
from tabu_lab.models.restoration_v54 import (
    V54Config,
    V54LossConfig,
    V54Model,
    prepare_episode,
    score_prepared_episode,
)


@pytest.fixture(autouse=True)
def deterministic():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(43)
        yield
    torch.set_num_threads(old)


def model():
    return V54Model(V54Config.from_size(
        "nano", backbone={"slots": 4, "ff_width": 16}, center_chunk_size=3,
    )).double()


def episode(column=None, *, damage=False):
    inputs, _, truth = example_episode()
    query = torch.zeros_like(inputs.query)
    query[-1, slice(None) if column is None else column] = True
    kwargs = {"replacements": {(0, 0): 9.0}} if damage else {}
    return make_episode(inputs.schema, truth.values, torch.ones_like(query), query,
                        code_seed=inputs.code_seed, **kwargs)


def same_nested(left, right):
    if isinstance(left, torch.Tensor):
        # Removing zero-weight rows changes GEMM reduction shapes. This is
        # numerical equivalence in FP64, not bitwise trajectory identity.
        torch.testing.assert_close(left, right, atol=1e-10, rtol=1e-8)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same_nested(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            same_nested(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("column", [None, 0, 1, 2])
@pytest.mark.parametrize("weights", [(0., 1., 0., 0.), (0., 1., 0., .2), None])
@pytest.mark.parametrize("keep_rows", [False, True])
def test_three_updates_match_full_readout_and_keep_all_centers(column, weights, keep_rows):
    reference = model()
    fast = copy.deepcopy(reference)
    optimizers = [torch.optim.AdamW(net.parameters(), lr=1e-4)
                  for net in (reference, fast)]
    args = episode(column, damage=True)
    loss_config = V54LossConfig(state_weights=weights)
    full = prepare_episode(reference, *args)
    selected = prepare_training_episode(fast, *args, loss_config, keep_zero_weight_rows=keep_rows)
    assert torch.equal(full.visible.inputs.visible, selected.visible.inputs.visible)
    assert torch.equal(full.visible.inputs.query, selected.visible.inputs.query)
    assert len(selected.visible.inputs.visible) == len(full.visible.inputs.visible)
    if weights is not None:
        expected = torch.tensor(weights)[full.states] != 0
        if keep_rows:
            active_columns = full.visible.request.targets[expected, 1].unique()
            expected = torch.isin(full.visible.request.targets[:, 1], active_columns)
        assert torch.equal(selected.visible.request.targets, full.visible.request.targets[expected])
    for _ in range(3):
        scores = []
        for net, opt, prepared in zip((reference, fast), optimizers, (full, selected), strict=True):
            opt.zero_grad(set_to_none=True)
            score = score_prepared_episode(net, prepared, loss_config)
            score.loss.backward()
            scores.append(score)
        same_nested(scores[0].loss, scores[1].loss)
        for column_result in scores[1].output.columns:
            a = column_result.column
            same_nested(scores[0].output.slopes[a], scores[1].output.slopes[a])
        for p, q in zip(reference.parameters(), fast.parameters(), strict=True):
            assert (p.grad is None) == (q.grad is None)
            if p.grad is not None:
                same_nested(p.grad, q.grad)
        for opt in optimizers:
            opt.step()
        same_nested(reference.state_dict(), fast.state_dict())
        same_nested(optimizers[0].state_dict(), optimizers[1].state_dict())


def test_zero_weight_columns_still_undergo_full_truth_and_support_admission():
    net = model()
    inputs, request, truth = episode(1)
    bad = list(truth.values)
    bad[0] = bad[0].clone()
    bad[0][0] = torch.nan  # No numeric Query, but cannot hide bad retained truth.
    with pytest.raises(ValueError, match="finite"):
        prepare_training_episode(net, inputs, request, TruthSidecar(tuple(bad), truth.states),
                                 V54LossConfig())
    constant = list(truth.values)
    constant[0] = torch.ones_like(constant[0])
    args = make_episode(inputs.schema, tuple(constant), torch.ones_like(inputs.visible),
                        inputs.query, code_seed=17)
    with pytest.raises(ValueError, match="distinct"):
        prepare_training_episode(net, *args, V54LossConfig())


def test_selected_snapshot_rejects_mutation_and_full_evaluation_is_unchanged():
    net = model()
    args = episode(1)
    selected = prepare_training_episode(net, *args, V54LossConfig())
    full = prepare_episode(net, *args)
    assert len(selected.visible.request.targets) == 1
    assert len(score_prepared_episode(net, full, decode=True).per_target) == 18
    selected.visible.inputs.values[0].add_(1)
    with pytest.raises(ValueError, match="mutated"):
        score_prepared_episode(net, selected)


@pytest.mark.parametrize("weights", [None, (1., 0., 0., 0.), (0., 1., 0., .1)])
def test_pruned_snapshot_cannot_silently_enable_omitted_loss_states(weights):
    net = model()
    selected = prepare_training_episode(net, *episode(1), V54LossConfig())
    with pytest.raises(ValueError, match="omitted readout states"):
        score_prepared_episode(net, selected, V54LossConfig(state_weights=weights))
    # Reweighting kept states within the execution contract is valid.
    ordinary = score_prepared_episode(net, selected)
    doubled = score_prepared_episode(net, selected, V54LossConfig(state_weights=(0., 2., 0., 0.)))
    torch.testing.assert_close(doubled.loss, 2 * ordinary.loss)
