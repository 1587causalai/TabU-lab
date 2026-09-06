"""Differentiable FP64 same-column Gaussian LL/NW. No learned terminal tensors."""

from __future__ import annotations

import torch

from .types import TARPrediction


def gaussian_weights(delta, bandwidth=1.0):
    logits = -delta.square().sum(-1) / (2 * bandwidth**2)
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("numerical-failure: nonfinite matching logits")
    return torch.softmax(logits, dim=-1)


def local_linear(delta, alpha, y, ridge):
    md = (alpha[..., None] * delta).sum(-2)
    my = (alpha * y).sum(-1)
    e = delta - md.unsqueeze(-2)
    cov = e.transpose(-1, -2) @ (alpha[..., None] * e)
    cov = (cov + cov.transpose(-1, -2)) / 2
    g = (alpha[..., None] * e * (y - my.unsqueeze(-1))[..., None]).sum(-2)
    a = cov + ridge * torch.eye(delta.shape[-1], device=delta.device, dtype=delta.dtype)
    chol, info = torch.linalg.cholesky_ex(a)
    if bool((info != 0).any()):
        raise FloatingPointError("numerical-failure: LL Cholesky failed; ridge is unchanged")
    slope = torch.cholesky_solve(g.unsqueeze(-1), chol).squeeze(-1)
    return my - (slope * md).sum(-1)


def predict_terminal(episode, z, stats, cfg):
    outputs = {}
    device = z.device
    visible = episode.visible.to(device)
    queries = episode.queries.to(device)
    values = episode.values.detach().to(device=device, dtype=torch.float64)
    for a, feature in enumerate(episode.features):
        supports = visible[:, a].nonzero().flatten()
        targets = queries[:, a].nonzero().flatten()
        count = len(supports)
        if count < cfg.minimum_support:
            for r in targets.tolist():
                outputs[(r, a)] = TARPrediction((r, a), "insufficient-support", count)
            continue
        support_z = z[supports, a].double()
        val = values[supports, a]
        mu, scale = stats[a]
        for ts in targets.split(cfg.terminal_query_chunk):
            if not len(ts):
                continue
            # Deliberately keep both target and support autograd paths.
            delta = support_z[None, :, :] - z[ts, a].double()[:, None, :]
            alpha = gaussian_weights(delta, cfg.match_bandwidth)
            if feature.kind == "numeric":
                y = (val - mu) / scale
                eta = local_linear(delta, alpha, y, cfg.ll_ridge)
                pred = mu + scale * eta
                for idx, r in enumerate(ts.tolist()):
                    if not bool(torch.isfinite(pred[idx])):
                        raise FloatingPointError("numerical-failure: nonfinite LL prediction")
                    outputs[(r, a)] = TARPrediction(
                        (r, a), "ok", count, pred[idx], scale=scale, weights=alpha[idx]
                    )
            else:
                domain = len(feature.domain)
                pmf = torch.zeros(len(ts), domain, device=device, dtype=torch.float64).scatter_add(
                    1, val.long()[None, :].expand(len(ts), -1), alpha
                )
                eps = cfg.category_smoothing_per_class
                pmf = (pmf + eps) / (1 + domain * eps)
                for idx, r in enumerate(ts.tolist()):
                    outputs[(r, a)] = TARPrediction(
                        (r, a), "ok", count, pmf[idx].argmax(), pmf[idx], weights=alpha[idx]
                    )
    return tuple(outputs[key] for key in sorted(outputs))
