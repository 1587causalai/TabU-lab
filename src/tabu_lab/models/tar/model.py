"""Truth-free TAR compiler, alternating dynamics, and typed additive response."""

from __future__ import annotations

import hashlib
import math

import torch
from torch import nn

from .attention import TAROMAB
from .config import TARConfig
from .terminal import predict_terminal
from .types import TAREpisode, TAROutput


class TARAxisBlock(nn.Module):
    def __init__(self, cfg, **kw):
        super().__init__()
        self.config = cfg
        if cfg.inducing_enabled:
            self.inducing = nn.Parameter(torch.empty(cfg.inducing_slots, cfg.width, **kw))
            self.collect = TAROMAB(cfg, **kw)
            self.read = TAROMAB(cfg, **kw)
        else:
            self.column = TAROMAB(cfg, **kw)
        self.row = TAROMAB(cfg, **kw)

    def forward(self, h, visible, null):
        # Address masks are compiled once per episode. Columns become an
        # independent batch dimension; no ragged source gathering is needed.
        columns = h.transpose(0, 1)
        source_mask, null_mask = visible.T, null.T
        cols = []
        for start in range(0, len(columns), self.config.collect_column_chunk):
            x = columns[start : start + self.config.collect_column_chunk]
            mask = source_mask[start : start + len(x)]
            nulls = null_mask[start : start + len(x)]
            if self.config.inducing_enabled:
                seeds = self.inducing.expand(len(x), -1, -1)
                summary = self.collect(seeds, x, source_mask=mask)
                eligible = (mask & (x != 0).any(-1)).any(-1)
                # Empty evidence cannot turn the inducing seed residual into facts.
                summary = torch.where(eligible[:, None, None], summary, 0)
                out = self.read(x, summary, null_mask=nulls)
            else:
                out = self.column(x, x, source_mask=mask, null_mask=nulls)
            cols.append(out)
        hc = torch.cat(cols, dim=0).transpose(0, 1)
        rows = []
        for start in range(0, len(hc), self.config.receiver_chunk_rows):
            x = hc[start : start + self.config.receiver_chunk_rows]
            rows.append(
                self.row(
                    x,
                    x,
                    source_mask=visible[start : start + len(x)],
                    null_mask=null[start : start + len(x)],
                )
            )
        return torch.cat(rows, dim=0)


class TabUTARModel(nn.Module):
    model_id = "tabu.tar"

    def __init__(self, config: TARConfig | None = None, *, device="cpu", dtype=torch.float32):
        super().__init__()
        self.config = cfg = config or TARConfig()
        if torch.device(device).type not in ("cpu", "cuda", "meta"):
            raise ValueError("TAR reference supports CPU/CUDA; FP64 terminal requires float64")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("TAR correctness backend supports float32/float64")
        kw = dict(device=device, dtype=dtype)
        d, k, j = cfg.width, cfg.semantic_slots, cfg.fourier_frequencies
        # Construct on CPU then move for CUDA so initialization is reproducible and
        # does not consume caller CUDA RNG. Meta construction is allocation-free.
        create_kw = dict(device="meta" if str(device) == "meta" else "cpu", dtype=dtype)
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(cfg.initialization_seed)
            self.frequencies = nn.Parameter(torch.empty(j, **create_kw))
            self.continuous = nn.Parameter(torch.empty(d, 2 * j, **create_kw))
            self.unit_query = nn.Parameter(torch.empty(k, d, **create_kw))
            self.feature_query = nn.Parameter(torch.empty(k, d, **create_kw))
            self.cell_query = nn.Parameter(torch.empty(d, **create_kw))
            self.response_cell = nn.Parameter(torch.empty(k, d, **create_kw))
            self.response_feature = nn.Parameter(torch.empty(d, **create_kw))
            self.response_unit = nn.Parameter(torch.empty(d, **create_kw))
            self.layers = nn.ModuleList([TARAxisBlock(cfg, **create_kw) for _ in range(cfg.blocks)])
            if str(device) != "meta":
                self._initialize()
        self.to(**kw)

    @torch.no_grad()
    def _initialize(self):
        cfg = self.config

        def seed_for(name):
            digest = hashlib.sha256(f"{cfg.initialization_seed}:{name}".encode()).digest()
            torch.default_generator.manual_seed(int.from_bytes(digest[:8], "little") % (2**63 - 1))

        branch_count = cfg.blocks * (3 if cfg.inducing_enabled else 2)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                seed_for(name)
                nn.init.xavier_uniform_(module.weight)
                module.bias.zero_()
                if name.endswith((".o", ".ffn2")):
                    module.weight.div_(math.sqrt(2 * branch_count))
        j = cfg.fourier_frequencies
        self.frequencies.copy_(2 ** torch.linspace(-3, 3, j, dtype=self.continuous.dtype))
        seed_for("continuous")
        q, r = torch.linalg.qr(torch.randn_like(self.continuous), mode="reduced")
        sign = torch.where(r.diagonal() >= 0, 1.0, -1.0)
        self.continuous.copy_(q * sign / math.sqrt(j))
        for name, p in self.named_parameters():
            if name in ("unit_query", "feature_query", "cell_query") or name.endswith(".inducing"):
                seed_for(name)
                p.normal_()
                p.div_(p.norm(dim=-1, keepdim=True))
            elif name in ("response_cell", "response_feature", "response_unit"):
                seed_for(name)
                p.normal_(std=1 / math.sqrt(cfg.semantic_slots * cfg.width))

    def compile_episode(self, episode: TAREpisode):
        episode.validate()
        cfg = self.config
        device, dtype = self.continuous.device, self.continuous.dtype
        if device.type == "meta":
            raise ValueError("meta model is for shape inspection only")
        # Detach input data at the declared non-learning preprocessing boundary.
        x = episode.values.detach().to(device=device, dtype=torch.float64)
        vis, queries = episode.visible.to(device), episode.queries.to(device)
        n, m = x.shape
        cols, stats, books, classes = [], [], {}, {}
        for a, feature in enumerate(episode.features):
            ids = vis[:, a].nonzero().flatten()
            vals = x[ids, a]
            mean = vals.mean() if len(vals) else x.new_zeros(())
            var = ((vals - mean) ** 2).mean() if len(vals) else x.new_zeros(())
            scale = (var + cfg.terminal_scale_floor**2).sqrt()
            stats.append((mean, scale))
            col = self.continuous.new_zeros(n, cfg.width)
            if len(vals):
                if feature.kind == "nominal":
                    cl = tuple(int(i) for i in vals.unique(sorted=True).tolist())
                    if feature.column_id in episode.codebooks:
                        cb = (
                            episode.codebooks[feature.column_id]
                            .detach()
                            .to(device=device, dtype=dtype)
                        )
                        supplied = episode.codebook_classes.get(feature.column_id)
                        if supplied != cl or cb.shape != (len(cl), cfg.width):
                            raise ValueError("invalid-input: codebook class/shape mismatch")
                        if not bool(torch.isfinite(cb).all()) or not torch.allclose(
                            cb.double().norm(dim=-1),
                            torch.ones(len(cl), device=device, dtype=torch.float64),
                            atol=1e-6,
                            rtol=1e-6,
                        ):
                            raise ValueError(
                                "invalid-input: nominal codes must be finite unit vectors"
                            )
                    else:
                        rows = []
                        for category in cl:
                            key = f"{episode.codebook_seed}:{feature.column_id}:{category}"
                            seed = int.from_bytes(
                                hashlib.sha256(key.encode()).digest()[:8], "little"
                            ) % (2**63 - 1)
                            gen = torch.Generator(device="cpu").manual_seed(seed)
                            g = torch.randn(cfg.width, generator=gen, dtype=torch.float64)
                            rows.append(g / g.norm())
                        cb = torch.stack(rows).to(device=device, dtype=dtype)
                    indices = torch.searchsorted(torch.tensor(cl, device=device), vals.long())
                    encoded = cb[indices]
                    books[feature.column_id], classes[feature.column_id] = cb, cl
                else:
                    if feature.kind == "numeric":
                        coordinate = (vals - mean) / (var.sqrt() + cfg.encoder_scale_epsilon)
                    else:
                        coordinate = vals / max(1, len(feature.domain) - 1)
                    phase = coordinate.to(dtype=dtype).unsqueeze(-1) * self.frequencies
                    encoded = torch.cat((phase.sin(), phase.cos()), dim=-1) @ self.continuous.T
                if not bool(torch.isfinite(encoded).all()) or bool((encoded == 0).all(-1).any()):
                    raise FloatingPointError("numerical-failure: malformed initial Value carrier")
                col = col.index_copy(0, ids, encoded)
            col = torch.where(queries[:, a, None], self.cell_query, col)
            cols.append(col)
        cell = torch.stack(cols, dim=1)
        units = self.unit_query[None, :, :].expand(n, -1, -1)
        features = self.feature_query[:, None, :].expand(-1, m, -1)
        corner = cell.new_zeros(cfg.semantic_slots, cfg.semantic_slots, cfg.width)
        h = torch.cat((torch.cat((cell, units), 1), torch.cat((features, corner), 1)), 0)
        v = torch.zeros(h.shape[:2], device=device, dtype=torch.bool)
        v[:n, :m] = vis
        null = torch.ones_like(v)
        null[:n, :m] = ~(vis | queries)
        null[:n, m:] = False
        null[n:, :m] = False
        if not bool(torch.isfinite(h).all()) or bool((h[~null] == 0).all(-1).any()):
            raise FloatingPointError("numerical-failure: malformed initial Query carrier")
        return h, v, null, stats, books, classes

    def forward(self, episode: TAREpisode) -> TAROutput:
        if not isinstance(episode, TAREpisode):
            raise TypeError("TAR forward accepts TAREpisode; truth is a separate scorer input")
        h, v, null, stats, books, classes = self.compile_episode(episode)
        n, m = episode.values.shape
        for layer in self.layers:
            h = layer(h, v, null)
        cfg = self.config
        cell = h[:n, :m] @ self.response_cell.T
        unit = (h[:n, m:] @ self.response_unit)[:, None, :]
        feature = (h[n:, :m] @ self.response_feature).T[None, :, :]
        z = cell + cfg.lambda_feature * feature + cfg.lambda_unit * unit
        if not bool(torch.isfinite(z).all()):
            raise FloatingPointError("numerical-failure: nonfinite response")
        predictions = predict_terminal(episode, z, stats, cfg)
        return TAROutput(predictions, h, z, books, classes)

    def forward_batch(self, episodes):
        """Ragged batch; no attention or statistics shared across episodes."""
        return tuple(self(ep) for ep in episodes)
