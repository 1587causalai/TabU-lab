"""Parameter-compatible target broadcasting for supervised rows.

The first V6 experiment retains every V5.5 model tensor.  It changes the
forward map and readout contract, so loading those tensors is weights-only
initialization, not a strict training resume.
"""

from __future__ import annotations

import torch

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
    """V5.5 tensors with broadcast; select the LL response without new weights.

    ``target_only`` is the default: original V5.5 per-column LL answers and
    Query-target loss. ``joint_all`` preserves the first V6 experiment, with
    ``e_a(x) + e_target(y)`` responses and all Query-row columns in the loss.
    """

    def __init__(self, config: V55Config, *, supervision: str = "target_only"):
        super().__init__(config)
        if config.slope_source != "shared_ll":
            raise ValueError("V6 first experiment requires shared_ll")
        if supervision not in ("target_only", "joint_all"):
            raise ValueError("supervision must be 'target_only' or 'joint_all'")
        self.supervision = supervision

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

        # Only the optional joint response needs target codes from visible
        # support labels. Query truth is never passed to this forward map.
        target_fact = facts[target]
        target_codes = None
        if self.supervision == "joint_all":
            target_codes = target_fact.answers.encoded.new_zeros(n, ANSWER_WIDTH)
            target_codes = target_codes.index_copy(0, target_fact.rows,
                                                   target_fact.answers.encoded)
        columns, slopes = [], {}
        for a, positions in enumerate(prepared.positions):
            if not len(positions):
                continue
            fact = facts[a]
            if self.supervision == "joint_all":
                complete = inputs.visible[fact.rows, target]
                support_rows = fact.rows[complete]
                answers = fact.answers.encoded[complete] + target_codes[support_rows]
            else:
                support_rows = fact.rows
                answers = fact.answers.encoded
            if not len(support_rows):
                columns.append(ColumnPrediction(
                    a, positions, EncodedRestoration("no-support", 0), None
                ))
                continue
            finite(answers, "V6 support answers")
            regression_cells = self.regression(h[:n, a])
            finite(regression_cells, "V6 regression Cell features")
            slope = shared_slope(
                units, support_rows, regression_cells[support_rows], answers,
                ridge=self.config.ridge, bandwidth=self.config.bandwidth,
                center_chunk_size=self.config.center_chunk_size,
            )
            slopes[a] = slope
            result = evaluate_column(
                units, regression_cells, support_rows, answers,
                request.targets[positions, 0], slope,
                bandwidth=self.config.bandwidth,
                chunk_size=self.config.center_chunk_size,
            )
            decoded = None
            if decode:
                if self.supervision == "target_only":
                    decoded = fact.answers.decode(result.encoding)
                elif a == target:
                    decoded = target_fact.answers.decode(result.encoding / 2)
            columns.append(ColumnPrediction(a, positions, result, decoded))
        return V53Output(request, tuple(columns), facts, h, units, slopes)
