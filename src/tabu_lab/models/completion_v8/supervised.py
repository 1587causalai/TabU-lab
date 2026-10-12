"""Support-row attention with mask-free Cell dynamics and exact Null closure.

Source sets are tensor slices: Support rows across rows; all real columns
within each row. Zero Cells remain in those sets. Unit/Feature receivers keep
their existing structural roles. No visible/query Cell eligibility is passed
to attention, and no intermediate Null reset repairs the dynamics.
"""

import torch

from ..restoration.backbone import AxialLayer

SUPPORT_ROWS_NULL = "support_rows_null_v1"


def supervised_rows(inputs):
    """Single-target supervised episodes: each row is labeled or a Query."""
    targets = inputs.query.any(0).nonzero().flatten()
    if len(targets) != 1:
        raise ValueError("support_rows_null_v1 requires exactly one supervised target column")
    target = int(targets[0])
    support = inputs.visible[:, target]
    query = inputs.query[:, target]
    if not bool((support | query).all()):
        raise ValueError("each supervised row must have a visible or Query target")
    return support


class SupervisedAxialLayer(AxialLayer):
    def forward(self, h, support_rows):
        n, m = h.shape[0] - 1, h.shape[1] - 1
        columns = h[:, :m].transpose(0, 1)
        # Exclude whole Query rows BEFORE any K/V projection or slot collect.
        sources = columns[:, :n][:, support_rows]
        if self.collect is not None:
            summaries, presence = self.collect._batched_with_presence(
                self.slot_seed.unsqueeze(0).expand(m, -1, -1), sources, None
            )
            # Slot seeds are not evidence. Zero Support carriers cannot wake
            # a slot, even if Query feature carriers in that column are nonzero.
            summaries = summaries * torch.isfinite(presence).any(-1)[:, None, None]
            sources = summaries
        real_columns = self.column.batched(columns, sources, None).transpose(0, 1)
        unit_column = self.column.batched(
            h[:, m:m + 1].transpose(0, 1), h.new_zeros(1, 0, h.shape[-1]), None
        )
        h = torch.cat((real_columns, unit_column.transpose(0, 1)), dim=1)

        # Every real column address participates. Null's zero presence and
        # bias-free value maps supply zero mass without a Cell source mask.
        sources = h[:n, :m]
        if self.row_collect is not None:
            summaries, presence = self.row_collect._batched_with_presence(
                self.row_slot_seed.unsqueeze(0).expand(n, -1, -1), sources, None
            )
            sources = summaries * torch.isfinite(presence).any(-1)[:, None, None]
        rows = self.row.batched(h[:n], sources, None)
        feature_row = self.row.batched(h[n:], h.new_zeros(1, 0, h.shape[-1]), None)
        return torch.cat((rows, feature_row), dim=0)


def install_supervised_layers(backbone):
    """Reuse the actual modules/parameters; no draws, copies or new weights."""
    axial = getattr(backbone, "axial", backbone)
    for layer in axial.layers:
        layer.__class__ = SupervisedAxialLayer


def supervised_dynamics(backbone, h, support_rows):
    axial = getattr(backbone, "axial", backbone)
    for layer in axial.layers:
        h = layer(h, support_rows)
    # Optional inherited Unit layers also exchange information across rows.
    n, m = h.shape[0] - 1, h.shape[1] - 1
    units = h[:n, m]
    for block in getattr(backbone, "unit_blocks", []):
        units = block.batched(units[None], units[support_rows][None], None)[0]
    if len(getattr(backbone, "unit_blocks", [])):
        rows = torch.arange(n, device=h.device)
        h = h.index_put((rows, torch.full_like(rows, m)), units)
    return h
