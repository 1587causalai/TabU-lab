"""Optional Cell-only execution of the inherited relational weights.

The old backbone is untouched. Under its existing source contract, Unit and
Feature are receivers only, so their computations cannot affect Cell outputs.
This alternate path retains all checkpoint keys but does not execute those
receivers or the dedicated Unit stack.
"""

import torch


def cell_only_dynamics(backbone, cells, sources, query):
    axial = backbone.axial if hasattr(backbone, "axial") else backbone
    valid = sources | query
    h = cells.masked_fill(~valid[..., None], 0)
    n, m = valid.shape
    for layer in axial.layers:
        columns = h.transpose(0, 1)
        eligible = sources.T
        if layer.collect is not None:
            summaries, presence = layer.collect._batched_with_presence(
                layer.slot_seed.unsqueeze(0).expand(m, -1, -1), columns, eligible
            )
            summaries = summaries * torch.isfinite(presence).any(-1).to(h.dtype)[:, None, None]
            h = layer.column.batched(
                columns, summaries, sources.new_ones(m, summaries.shape[1])
            ).transpose(0, 1)
        else:
            h = layer.column.batched(columns, columns, eligible).transpose(0, 1)
        h = h.masked_fill(~valid[..., None], 0)
        if layer.row_collect is not None:
            summaries, presence = layer.row_collect._batched_with_presence(
                layer.row_slot_seed.unsqueeze(0).expand(n, -1, -1), h, sources
            )
            summaries = summaries * torch.isfinite(presence).any(-1).to(h.dtype)[:, None, None]
            h = layer.row.batched(h, summaries, sources.new_ones(n, summaries.shape[1]))
        else:
            h = layer.row.batched(h, h, sources)
        h = h.masked_fill(~valid[..., None], 0)
    return h
