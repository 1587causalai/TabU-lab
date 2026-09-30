"""Opt-in V5.5 all-observation restoration after the V6 broadcast forward.

No changes to legacy models, target-only or joint-all scorers. Visible cells
retain the original self-support semantics. Query truth remains scorer-only.
"""
from tabu_lab.models.restoration_v53.training import (
    V53LossConfig, prepare_training_episode, score_prepared_episode,
)
from tabu_lab.models.restoration_v6.training import V6Score
from tabu_lab.models.restoration_v6.model import task_target_column


def score_training_episode(model, inputs, truth, *, request, loss_config):
    assert model.supervision == 'target_only'  # Own-cell LL responses, with broadcast.
    assert loss_config.state_weights is None
    target = task_target_column(inputs)
    prepared = prepare_training_episode(model, inputs, request, truth, loss_config)
    count = int((truth.states >= 0).sum())
    assert len(prepared.visible.request.targets) == count
    score = score_prepared_episode(model, prepared, loss_config, decode=False)
    assert len(score.per_target) == count
    return V6Score(score.loss, score.per_target, score.output,
                   int(inputs.query[:, target].sum()), count)
