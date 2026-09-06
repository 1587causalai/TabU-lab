"""TAR rectangular OMAB, with exact eligibility and globally normalized chunking."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def presence(x, tau=1.0):
    # FP64 avoids squaring underflow/overflow for finite FP32 carriers. Never use
    # 1-tau/(tau+n): that loses small receivers to catastrophic cancellation.
    work = x.double()
    scale = work.abs().amax(-1).clamp_min(1)
    norm = (work / scale.unsqueeze(-1)).square().sum(-1)
    return (norm / ((tau / scale) / scale + norm)).to(x.dtype)


def log_presence(x, tau=1.0):
    # Called only for exact nonzero sources. Work in log space; no source threshold.
    work = x.double()
    maximum = work.abs().amax(-1)
    log_norm = 2 * maximum.log() + (work / maximum.unsqueeze(-1)).square().sum(-1).log()
    return (-F.softplus(math.log(tau) - log_norm)).to(x.dtype)


class SerialOMAB(nn.Module):
    def __init__(self, config, *, device=None, dtype=None):
        super().__init__()
        self.config = config
        kw = dict(device=device, dtype=dtype)
        d = config.width
        for name in ("q", "k", "v", "o"):
            setattr(self, name, nn.Linear(d, d, **kw))
        self.gain = nn.Parameter(torch.ones(d, **kw))
        self.ffn1 = nn.Linear(d, config.ff_width, **kw)
        self.ffn2 = nn.Linear(config.ff_width, d, **kw)

    def forward(self, receivers, sources, *, null_mask=None, source_mask=None):
        if receivers.ndim != 2 or sources.ndim != 2:
            raise ValueError("OMAB expects [receivers,d] and [sources,d]")
        if not bool(torch.isfinite(receivers).all() & torch.isfinite(sources).all()):
            raise FloatingPointError("numerical-failure: nonfinite OMAB input")
        cfg = self.config
        if null_mask is None:
            null_mask = torch.zeros(len(receivers), dtype=torch.bool, device=receivers.device)
        ids = (~null_mask).nonzero().flatten()
        if not len(ids):
            return torch.zeros_like(receivers)
        # Source eligibility is independent of projected K/V or continuous mass.
        active = (sources != 0).any(-1)
        if source_mask is not None:
            active = active & source_mask
        src = sources[active]
        outputs = []
        h, dh = cfg.heads, cfg.width // cfg.heads
        # Cache projected source chunks in the autograd graph, never detach.
        chunks = []
        for part in src.split(cfg.source_chunk):
            if len(part):
                chunks.append(
                    (
                        self.k(part).view(-1, h, dh).transpose(0, 1),
                        self.v(part).view(-1, h, dh).transpose(0, 1),
                        log_presence(part, cfg.presence_tau),
                    )
                )
        for rid in ids.split(cfg.receiver_chunk_rows):
            x = receivers[rid]
            if chunks:
                q = self.q(x).view(-1, h, dh).transpose(0, 1)
                state = None
                for k, v, lp in chunks:
                    logits = (q @ k.transpose(-1, -2)) / math.sqrt(dh) + lp
                    # Max shift has no mathematical derivative contribution.
                    m = logits.amax(-1, keepdim=True).detach()
                    w = (logits - m).exp()
                    z, u = w.sum(-1, keepdim=True), w @ v
                    if state is not None:
                        m0, z0, u0 = state
                        maximum = torch.maximum(m0, m)
                        a, b = (m0 - maximum).exp(), (m - maximum).exp()
                        z, u, m = a * z0 + b * z, a * u0 + b * u, maximum
                    state = m, z, u
                _, z, u = state
                mix = (u / z).transpose(0, 1).reshape(-1, cfg.width)
                update = presence(x, cfg.presence_tau).unsqueeze(-1) * self.o(mix)
            else:
                update = torch.zeros_like(x)  # Includes output bias deletion.
            r = x + update
            rms = r * torch.rsqrt(r.square().mean(-1, keepdim=True) + cfg.rms_epsilon) * self.gain
            y = r + presence(r, cfg.presence_tau).unsqueeze(-1) * self.ffn2(
                F.gelu(self.ffn1(rms), approximate="none")
            )
            if not bool(torch.isfinite(y).all()):
                raise FloatingPointError("numerical-failure: nonfinite OMAB output")
            outputs.append(y)
        return torch.zeros_like(receivers).index_copy(0, ids, torch.cat(outputs))


class SerialAxisBlock(nn.Module):
    def __init__(self, cfg, **kw):
        super().__init__()
        self.config = cfg
        if cfg.inducing_enabled:
            self.inducing = nn.Parameter(torch.empty(cfg.inducing_slots, cfg.width, **kw))
            self.collect = SerialOMAB(cfg, **kw)
            self.read = SerialOMAB(cfg, **kw)
        else:
            self.column = SerialOMAB(cfg, **kw)
        self.row = SerialOMAB(cfg, **kw)

    def forward(self, h, visible, null):
        cols = []
        for a in range(h.shape[1]):
            sources = h[visible[:, a], a]
            if self.config.inducing_enabled:
                if bool((sources != 0).any()):
                    summary = self.collect(self.inducing, sources)
                else:
                    summary = torch.zeros_like(self.inducing)
                col = self.read(h[:, a], summary, null_mask=null[:, a])
            else:
                col = self.column(h[:, a], sources, null_mask=null[:, a])
            cols.append(col)
        hc = torch.stack(cols, dim=1)
        return torch.stack(
            [self.row(hc[r], hc[r, visible[r]], null_mask=null[r]) for r in range(len(hc))]
        )
