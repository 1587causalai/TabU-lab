"""Independent, parameter-compatible switches for the first V6 mechanisms."""
from __future__ import annotations

import torch

from tabu_lab.models.restoration._validation import finite
from tabu_lab.models.restoration.model import ColumnPrediction
from tabu_lab.models.restoration.readout import EncodedRestoration
from tabu_lab.models.restoration_v53.encoding import ANSWER_WIDTH
from tabu_lab.models.restoration_v53.model import PreparedV53, V53Output
from tabu_lab.models.restoration_v53.readout import evaluate_column, shared_slope
from tabu_lab.models.restoration_v55 import V55Model
from tabu_lab.models.restoration_v6.model import task_target_column


VARIANTS = {
    "broadcast-only": dict(broadcast=True, joint_answers=False),
    "all-column-loss-only": dict(broadcast=False, joint_answers=True),
}


class MechanismAblation(V55Model):
    """Same tensors; select either input broadcast or joint-response supervision.

    broadcast-only: h0=Broadcast(hbase), LL response=e_a, old target-only loss.
    all-column-loss-only: h0=hbase, LL response=e_a+e_target, V6 all-column loss.
    Loss is selected by the isolated runner; this class implements forward only.
    """

    def __init__(self, config, variant):
        super().__init__(config)
        if config.slope_source != "shared_ll" or variant not in VARIANTS:
            raise ValueError("unsupported ablation or non-shared LL configuration")
        self.variant = variant
        self.use_broadcast = VARIANTS[variant]["broadcast"]
        self.joint_answers = VARIANTS[variant]["joint_answers"]

    def forward_prepared(self, prepared: PreparedV53, *, decode=True) -> V53Output:
        if prepared.encoder_id != id(self.encoder) or prepared.config != self.config:
            raise ValueError("prepared ablation model changed")
        prepared.versions.validate()
        inputs, request, facts = prepared.inputs, prepared.request, prepared.facts
        target = task_target_column(inputs)
        n, m = inputs.visible.shape
        base = self.encoder.forward_prepared(inputs, prepared.features)
        if self.use_broadcast:
            active = inputs.visible | inputs.query
            cells = torch.where(active[..., None],
                                base[:n, :m] + base[:n, target][:, None, :],
                                torch.zeros_like(base[:n, :m]))
            h = torch.cat((torch.cat((cells, base[:n, m:m + 1]), dim=1), base[n:]), dim=0)
        else:
            h = base
        finite(h, "ablation initial carriers")
        h = self.backbone(h, inputs.visible, inputs.query)
        units = h[:n, m]
        eligible = inputs.visible.any(-1)
        for block in self.unit_blocks:
            units = block(units, units, eligible)

        target_fact = facts[target]
        target_codes = None
        if self.joint_answers:
            target_codes = target_fact.answers.encoded.new_zeros(n, ANSWER_WIDTH)
            target_codes = target_codes.index_copy(0, target_fact.rows,
                                                   target_fact.answers.encoded)
        columns, slopes = [], {}
        for a, positions in enumerate(prepared.positions):
            if not len(positions):
                continue
            fact = facts[a]
            if self.joint_answers:
                complete = inputs.visible[fact.rows, target]
                support_rows = fact.rows[complete]
                answers = fact.answers.encoded[complete] + target_codes[support_rows]
            else:
                # Restore the original response and support rules exactly.
                support_rows, answers = fact.rows, fact.answers.encoded
            if not len(support_rows):
                columns.append(ColumnPrediction(a, positions,
                                                 EncodedRestoration("no-support", 0), None))
                continue
            finite(answers, "ablation support answers")
            cells = self.regression(h[:n, a])
            finite(cells, "ablation regression Cell features")
            slope = shared_slope(units, support_rows, cells[support_rows], answers,
                                 ridge=self.config.ridge, bandwidth=self.config.bandwidth,
                                 center_chunk_size=self.config.center_chunk_size)
            slopes[a] = slope
            result = evaluate_column(units, cells, support_rows, answers,
                                     request.targets[positions, 0], slope,
                                     bandwidth=self.config.bandwidth,
                                     chunk_size=self.config.center_chunk_size)
            decoded = None
            if decode:
                if self.joint_answers:
                    if a == target:
                        decoded = target_fact.answers.decode(result.encoding / 2)
                else:
                    decoded = fact.answers.decode(result.encoding)
            columns.append(ColumnPrediction(a, positions, result, decoded))
        return V53Output(request, tuple(columns), facts, h, units, slopes)
