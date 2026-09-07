"""TAR rectangular OMAB, with exact eligibility and globally normalized chunking."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def presence(x, tau=1.0, *, fp32=False):
    # FP64 avoids squaring underflow/overflow for finite FP32 carriers. Never use
    # 1-tau/(tau+n): that loses small receivers to catastrophic cancellation.
    work = x.float() if fp32 else x.double()
    scale = work.abs().amax(-1).clamp_min(1)
    norm = (work / scale.unsqueeze(-1)).square().sum(-1)
    return (norm / ((tau / scale) / scale + norm)).to(x.dtype)


def log_presence(x, tau=1.0, *, fp32=False):
    # Called only for exact nonzero sources. Work in log space; no source threshold.
    work = x.float() if fp32 else x.double()
    maximum = work.abs().amax(-1)
    log_norm = 2 * maximum.log() + (work / maximum.unsqueeze(-1)).square().sum(-1).log()
    return (-F.softplus(math.log(tau) - log_norm)).to(x.dtype)


class TAROMAB(nn.Module):
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
        """Independent attention groups [..., R, d] and [..., S, d].

        Episode masks select sources/receivers without dynamic gathers. Batch
        dimensions never share normalization or evidence. Chunking only bounds
        the score workspace and merges a single global softmax per receiver.
        """
        if (
            receivers.ndim < 2
            or sources.ndim != receivers.ndim
            or receivers.shape[:-2] != sources.shape[:-2]
            or receivers.shape[-1] != self.config.width
            or sources.shape[-1] != self.config.width
        ):
            raise ValueError("OMAB expects matching [..., receivers,d] and [..., sources,d]")
        if not bool(torch.isfinite(receivers).all() & torch.isfinite(sources).all()):
            raise FloatingPointError("numerical-failure: nonfinite OMAB input")
        cfg = self.config
        if null_mask is None:
            null_mask = torch.zeros(receivers.shape[:-1], dtype=torch.bool, device=receivers.device)
        for mask, shape in ((null_mask, receivers.shape[:-1]), (source_mask, sources.shape[:-1])):
            if mask is not None and (
                mask.shape != shape or mask.dtype != torch.bool or mask.device != receivers.device
            ):
                raise ValueError("OMAB masks must be boolean and match their address axes")
        if receivers.shape[-2] == 0:
            return torch.zeros_like(receivers)
        active = (sources != 0).any(-1)
        if source_mask is not None:
            active = active & source_mask
        # Excluded data does not participate in projections or their gradients.
        src = torch.where(active.unsqueeze(-1), sources, 0)
        has_source = active.any(-1)[..., None, None]
        h, dh = cfg.heads, cfg.width // cfg.heads

        def heads(x):
            return x.unflatten(-1, (h, dh)).transpose(-3, -2)

        chunks = []
        for start in range(0, src.shape[-2], cfg.source_chunk):
            part = src[..., start : start + cfg.source_chunk, :]
            eligible = active[..., start : start + cfg.source_chunk]
            # Inactive entries use a finite placeholder ONLY in log_presence;
            # exact eligibility removes them before softmax, including gradients.
            lp = log_presence(
                torch.where(eligible[..., None], part, 1),
                cfg.presence_tau,
                fp32=cfg.numerical_backend == "experimental_fp32",
            )
            chunks.append((heads(self.k(part)), heads(self.v(part)), lp, eligible))
        outputs = []
        for start in range(0, receivers.shape[-2], cfg.receiver_chunk_rows):
            null = null_mask[..., start : start + cfg.receiver_chunk_rows]
            x = torch.where(
                null[..., None], 0, receivers[..., start : start + cfg.receiver_chunk_rows, :]
            )
            if chunks:
                q = heads(self.q(x))
                state = None
                for k, v, lp, eligible in chunks:
                    logits = (q @ k.transpose(-1, -2)) / math.sqrt(dh) + lp[..., None, None, :]
                    logits = logits.masked_fill(~eligible[..., None, None, :], -torch.inf)
                    maximum = logits.amax(-1, keepdim=True).detach()
                    # Avoid -inf - -inf and NaN gradients for an empty group/chunk.
                    m = torch.where(torch.isfinite(maximum), maximum, 0)
                    w = (logits - m).exp()
                    z, u = w.sum(-1, keepdim=True), w @ v
                    if state is not None:
                        m0, z0, u0 = state
                        maximum = torch.where(
                            z0 == 0, m, torch.where(z == 0, m0, torch.maximum(m0, m))
                        )
                        a = torch.where(z0 > 0, m0 - maximum, 0).exp()
                        b = torch.where(z > 0, m - maximum, 0).exp()
                        z, u, m = a * z0 + b * z, a * u0 + b * u, maximum
                    state = m, z, u
                _, z, u = state
                mix = (u / z.clamp_min(torch.finfo(z.dtype).tiny)).transpose(-3, -2).flatten(-2)
                # No sources means zero attention update, including output bias.
                update = torch.where(has_source, self.o(mix), 0)
                update = (
                    presence(
                        x, cfg.presence_tau, fp32=cfg.numerical_backend == "experimental_fp32"
                    ).unsqueeze(-1)
                    * update
                )
            else:
                update = torch.zeros_like(x)
            r = x + update
            rms = r * torch.rsqrt(r.square().mean(-1, keepdim=True) + cfg.rms_epsilon) * self.gain
            y = r + presence(
                r, cfg.presence_tau, fp32=cfg.numerical_backend == "experimental_fp32"
            ).unsqueeze(-1) * self.ffn2(F.gelu(self.ffn1(rms), approximate="none"))
            outputs.append(torch.where(null[..., None], 0, y))
        output = torch.cat(outputs, dim=-2)
        if not bool(torch.isfinite(output).all()):
            raise FloatingPointError("numerical-failure: nonfinite OMAB output")
        return output
