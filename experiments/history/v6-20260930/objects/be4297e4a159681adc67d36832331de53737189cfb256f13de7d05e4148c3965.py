"""Parameter-compatible target broadcasting and joint-answer LL for supervised rows.

The first V6 experiment retains every V5.5 model tensor.  It changes the
forward map and readout contract, so loading those tensors is weights-only
initialization, not a strict training resume.
"""

from __future__ import annotations

import torch

from ..restoration._dtype import solve_dtype
from ..restoration._validation import finite
from ..restoration.model import ColumnPrediction
from ..restoration.readout import EncodedRestoration
from ..restoration_v53.encoding import ANSWER_WIDTH
from ..restoration_v53.model import PreparedV53, V53Output
from ..restoration_v53.readout import evaluate_column, shared_slope
from ..restoration_v55 import V55Config, V55Model


def task_target_column(inputs) -> int:
    """A supervised episode has one hidden target column and known support labels."""
    query_columns = inputs.query.any(0).nonzero(as_tuple=True)[0]
    if len(query_columns) != 1:
        raise ValueError("V6 requires exactly one Query target column")
    target = int(query_columns[0])
    if bool(inputs.query[:, ~torch.nn.functional.one_hot(
        query_columns, num_classes=inputs.query.shape[1]
    ).bool().squeeze(0)].any()):
        raise ValueError("V6 forbids Query cells outside the task target")
    if not bool((inputs.visible[:, target] | inputs.query[:, target]).all()):
        raise ValueError("V6 requires target labels visible on every support row")
    return target


class V6Model(V55Model):
    """V5.5 tensors, with one synchronous broadcast and all-column joint answers."""

    def __init__(self, config: V55Config):
        super().__init__(config)
        if config.slope_source != "shared_ll":
            raise ValueError("V6 first experiment requires shared_ll")

    def forward_prepared(self, prepared: PreparedV53, *, decode: bool = True) -> V53Output:
        if prepared.encoder_id != id(self.encoder) or prepared.config != self.config:
            raise ValueError("prepared V6 model changed")
        prepared.versions.validate()
        inputs, request, facts = prepared.inputs, prepared.request, prepared.facts
        target = task_target_column(inputs)
        n, m = inputs.visible.shape

        base = self.encoder.forward_prepared(inputs, prepared.features)
        active = inputs.visible | inputs.query
        cells = torch.where(
            active[..., None],
            base[:n, :m] + base[:n, target][:, None, :],
            torch.zeros_like(base[:n, :m]),
        )
        h = torch.cat((torch.cat((cells, base[:n, m:m + 1]), dim=1), base[n:]), dim=0)
        finite(h, "V6 broadcast carriers")
        h = self.backbone(h, inputs.visible, inputs.query)
        units = h[:n, m]
        eligible = inputs.visible.any(-1)
        for block in self.unit_blocks:
            units = block(units, units, eligible)

        # The target codec is fixed by support labels.  Query truth is absent
        # from this function and cannot become an LL answer or input carrier.
        target_fact = facts[target]
        target_codes = target_fact.answers.encoded.new_zeros(n, ANSWER_WIDTH)
        target_codes = target_codes.index_copy(0, target_fact.rows,
                                               target_fact.answers.encoded)
        columns, slopes = [], {}
        for a, positions in enumerate(prepared.positions):
            if not len(positions):
                continue
            fact = facts[a]
            complete = inputs.visible[fact.rows, target]
            support_rows = fact.rows[complete]
            if not len(support_rows):
                columns.append(ColumnPrediction(
                    a, positions, EncodedRestoration("no-support", 0), None
                ))
                continue
            joint_answers = fact.answers.encoded[complete] + target_codes[support_rows]
            finite(joint_answers, "V6 complete support answers")
            regression_cells = self.regression(h[:n, a])
            finite(regression_cells, "V6 regression Cell features")
            slope = shared_slope(
                units, support_rows, regression_cells[support_rows], joint_answers,
                ridge=self.config.ridge, bandwidth=self.config.bandwidth,
                center_chunk_size=self.config.center_chunk_size,
            )
            slopes[a] = slope
            result = evaluate_column(
                units, regression_cells, support_rows, joint_answers,
                request.targets[positions, 0], slope,
                bandwidth=self.config.bandwidth,
                chunk_size=self.config.center_chunk_size,
            )
            decoded = None
            if decode and a == target:
                decoded = target_fact.answers.decode(result.encoding / 2)
            columns.append(ColumnPrediction(a, positions, result, decoded))
        return V53Output(request, tuple(columns), facts, h, units, slopes)
