"""Bounded correctness probes, not fitting runs or issued research receipts."""

from __future__ import annotations

import copy
import io

import torch

from .backbone import BackboneConfig
from .contracts import ColumnSchema, make_episode
from .encoding import EncoderConfig
from .model import RestorationConfig, RestorationModel
from .training import score_episode


def example_episode(*, damage=False):
    schema = (
        ColumnSchema("measurement", "numeric"),
        ColumnSchema("category", "nominal", 3),
        ColumnSchema("rank", "ordinal", 3),
    )
    values = (
        torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64),
        torch.tensor([0, 1, 2, 0, 1, 2]),
        torch.tensor([0, 1, 2, 0, 1, 2]),
    )
    observed = torch.ones(6, 3, dtype=torch.bool)
    query = torch.zeros_like(observed)
    query[-1] = True
    kwargs = {}
    if damage:
        null = torch.zeros_like(observed)
        null[4, 0] = True
        kwargs = {"null": null, "replacements": {(0, 0): 9.0}}
    return make_episode(schema, values, observed, query, code_seed=17, **kwargs)


def small_config(kind="inducing", category_map="identity128", readout="ll"):
    return RestorationConfig(
        encoder=EncoderConfig(category_map=category_map, mlp_hidden=24),
        backbone=BackboneConfig(kind=kind, layers=1, ff_width=32, slots=4),
        readout=readout,
    )


def check_model_variants():
    counts = {}
    with torch.random.fork_rng(devices=[]):
        for kind in ("direct", "inducing"):
            for mapping in ("identity128", "rotary32", "mlp32", "mlp256"):
                for mode in ("ll", "nw"):
                    torch.manual_seed(41)
                    model = RestorationModel(small_config(kind, mapping, mode)).double()
                    score = score_episode(model, *example_episode(damage=True))
                    score.loss.backward()
                    gradients = [p.grad for p in model.parameters() if p.grad is not None]
                    assert gradients and all(bool(torch.isfinite(g).all()) for g in gradients)
                    assert sum(float(g.square().sum()) for g in gradients) > 0
                    feature = model.encoder.feature_seed.grad
                    assert feature is None or torch.count_nonzero(feature) == 0
                    if mode == "nw":
                        cell = model.encoder.cell_seed.grad
                        assert cell is None or torch.count_nonzero(cell) == 0
                    assert torch.count_nonzero(score.output.carriers[4, 0]) == 0
                    assert score.by_state["null"]["count"] == 1
                    counts[f"{kind}/{mapping}/{mode}"] = {
                        "parameters": sum(p.numel() for p in model.parameters()),
                        "loss": float(score.loss.detach()),
                    }
    return counts


def check_optimizer_continuation():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(101)
        model = RestorationModel(small_config()).double()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        episode = example_episode()

        def update(model, optimizer):
            optimizer.zero_grad(set_to_none=True)
            score = score_episode(model, *episode)
            score.loss.backward()
            optimizer.step()
            return score

        initial = copy.deepcopy(model.state_dict())
        update(model, optimizer)
        assert any(not torch.equal(initial[k], v) for k, v in model.state_dict().items())
        buffer = io.BytesIO()
        torch.save(
            {
                "config": model.config.as_dict(),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": torch.get_rng_state(),
            },
            buffer,
        )
        buffer.seek(0)
        saved = torch.load(buffer, weights_only=True)
        resumed = RestorationModel(RestorationConfig.from_dict(saved["config"])).double()
        resumed.load_state_dict(saved["model"], strict=True)
        resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-4)
        resumed_optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["rng"])
        left = update(model, optimizer)
        torch.set_rng_state(saved["rng"])
        right = update(resumed, resumed_optimizer)
        torch.testing.assert_close(left.loss, right.loss, rtol=0, atol=0)
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, resumed.state_dict()[name], rtol=0, atol=0)
        for key, state in optimizer.state_dict()["state"].items():
            for name, value in state.items():
                torch.testing.assert_close(
                    value, resumed_optimizer.state_dict()["state"][key][name], rtol=0, atol=0
                )
    return {"updates": 2, "model_optimizer_continuation": "exact", "checkpoint": "in-memory"}
