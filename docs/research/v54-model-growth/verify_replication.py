"""Forward-only algebra check: exact isometric widening of current V54Model.

No training, optimizer, checkpoint conversion, or remote host operation.
Writes one explicitly named, fresh local verification receipt.
The tiny model uses explicit dimensions, not a claim about a named size preset.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import sys
import json
import math
import time
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

import torch
from tabu_lab.models.restoration.backbone import BackboneConfig, OMAB
from tabu_lab.models.restoration.contracts import ColumnSchema, make_episode
from tabu_lab.models.restoration.readout import unit_kernel_logits
from tabu_lab.models.restoration_v54 import V54Config, V54Model
from tabu_lab.models.restoration_v53.readout import shared_slope, evaluate_column

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
torch.manual_seed(20260922)


def lift(x, k):
    return x.repeat(*([1] * (x.ndim - 1)), k) / math.sqrt(k)


def transform_block(old, new, k):
    for name in ('attention_presence', 'ff_presence', 'q', 'k', 'v', 'out'):
        original = getattr(old, name).weight
        scale = math.sqrt(k) if name in ('q', 'k') else 1.0
        getattr(new, name).weight.copy_(scale * torch.block_diag(*([original] * k)))
    new.norm_scale.copy_(old.norm_scale.repeat(k))
    new.ff[0].weight.copy_(torch.block_diag(*([old.ff[0].weight] * k)))
    new.ff[2].weight.copy_(torch.block_diag(*([old.ff[2].weight] * k)) / math.sqrt(k))


def transform_model(old, k):
    b = old.config.backbone
    cfg = replace(old.config, backbone=replace(
        b, width=k*b.width, heads=k*b.heads, ff_width=k*b.ff_width,
        norm_eps=b.norm_eps/k,
    ))
    new = V54Model(cfg).double().eval()
    new.encoder.projection.weight.copy_(old.encoder.projection.weight.repeat(k, 1) / math.sqrt(k))
    for name in ('cell_seed', 'unit_seed', 'feature_seed'):
        getattr(new.encoder, name).copy_(lift(getattr(old.encoder, name), k))
    for a, c in zip(old.backbone.layers, new.backbone.layers):
        for name in ('row', 'column'):
            transform_block(getattr(a, name), getattr(c, name), k)
        if a.collect is not None:
            transform_block(a.collect, c.collect, k)
            c.slot_seed.copy_(lift(a.slot_seed, k))
    for a, c in zip(old.unit_blocks, new.unit_blocks):
        transform_block(a, c, k)
    assert old.config.regression_width is None
    return new


def err(a, b):
    return (a - b).abs().max().item()


schema = (
    ColumnSchema('numeric', 'numeric'),
    ColumnSchema('nominal', 'nominal', 3),
    ColumnSchema('ordinal', 'ordinal', 4),
)
values = (
    torch.tensor([-2., -.5, .1, 1.2, 2., 3.4]),
    torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long),
    torch.tensor([0, 1, 2, 3, 0, 1], dtype=torch.long),
)
observed = torch.ones(6, 3, dtype=torch.bool)
observed[4, 0] = False
query = torch.zeros_like(observed)
query[5, :] = True
inputs, request, _ = make_episode(schema, values, observed, query, code_seed=41)
start = time.perf_counter()
records = []
with torch.no_grad():
    for kind in ('direct', 'inducing'):
        cfg = V54Config(
            size='nano',
            backbone=BackboneConfig(width=128, layers=1, heads=4,
                                    ff_width=128, slots=4, kind=kind),
            unit_layers=1,
        )
        old = V54Model(cfg).double().eval()
        # Nonidentity presence and nonconstant learned scales exercise the map
        # beyond identity initialization, without any optimizer/training step.
        for mod in old.modules():
            if hasattr(mod, 'attention_presence'):
                mod.attention_presence.weight.add_(torch.randn_like(mod.attention_presence.weight) * .02)
                mod.ff_presence.weight.add_(torch.randn_like(mod.ff_presence.weight) * .02)
                mod.norm_scale.copy_(torch.rand_like(mod.norm_scale) + .5)
        before = old(inputs, request)
        for k in (2, 3):
            new = transform_model(old, k)
            after = new(inputs, request)
            row = {
                'kind': kind, 'k': k, 'source_config': old.config.as_dict(),
                'destination_config': new.config.as_dict(),
                'parameter_count_original': sum(p.numel() for p in old.parameters()),
                'parameter_count_widened': sum(p.numel() for p in new.parameters()),
                'carrier_max_error': err(after.carriers, lift(before.carriers, k)),
                'unit_max_error': err(after.units, lift(before.units, k)),
                'squared_unit_distance_max_error': err(
                    torch.cdist(before.units, before.units).square(),
                    torch.cdist(after.units, after.units).square()),
                'kernel_max_error': err(
                    unit_kernel_logits(before.units, before.units, bandwidth=cfg.bandwidth),
                    unit_kernel_logits(after.units, after.units, bandwidth=cfg.bandwidth)),
                'shared_slope_max_error': max(err(after.slopes[a], lift(v, k))
                                              for a, v in before.slopes.items()),
                'encoding_max_error': max(err(a.result.encoding, b.result.encoding)
                                          for a, b in zip(before.columns, after.columns)),
                'decoded_numeric_max_error': err(before.columns[0].decoded, after.columns[0].decoded),
                'decoded_nominal_equal': torch.equal(before.columns[1].decoded, after.columns[1].decoded),
                'decoded_ordinal_equal': torch.equal(before.columns[2].decoded, after.columns[2].decoded),
            }
            checks = [value < 1e-10 for key, value in row.items() if key.endswith('_max_error')]
            row['passed'] = all(checks) and row['decoded_nominal_equal'] and row['decoded_ordinal_equal']
            records.append(row)

with torch.no_grad():
    # A legal OMAB with zero closure need not be idempotent.
    cfg = BackboneConfig(width=128, heads=4, ff_width=256, slots=4)
    block = OMAB(cfg).double().eval()
    block.q.weight.zero_()
    block.k.weight.zero_()
    block.v.weight.copy_(torch.eye(128))
    block.out.weight.copy_(torch.eye(128))
    block.ff[2].weight.zero_()
    x = torch.zeros(1, 128)
    x[0, 0] = 1.
    once = block(x, x, torch.ones(1, dtype=torch.bool))
    twice = block(once, once, torch.ones(1, dtype=torch.bool))
    p = x.square().sum() / (cfg.tau_presence + x.square().sum())
    expected = (1 + p.square() / (cfg.reference_mass + p)) * x
    assert err(once, expected) < 1e-12
    assert err(twice, once) > .1

    # Changing Unit geometry changes final LL/NW answers as well.
    unit_block = OMAB(cfg).double().eval()
    unit_block.out.weight.zero_()
    unit_block.ff[0].weight.copy_(torch.cat((torch.eye(128), -torch.eye(128)), 0))
    unit_block.ff[2].weight.copy_(torch.cat((torch.eye(128), -torch.eye(128)), 1))
    units = torch.zeros(3, 128)
    a = .2
    units[:, 0] = torch.tensor([-a, a, a])
    grown_units = unit_block(units, units, torch.ones(3, dtype=torch.bool))
    cells = torch.zeros(3, 128)
    supports = torch.tensor([0, 1])
    targets = torch.tensor([2])
    q, b = torch.zeros(128), torch.zeros(128)
    q[:4], b[4:8] = 1., 1.
    answers = torch.stack((q-b, q+b))
    predictions = []
    for geometry in (units, grown_units):
        slope = shared_slope(geometry, supports, cells[supports], answers,
                             ridge=.001, bandwidth=1., center_chunk_size=4)
        encoded = evaluate_column(geometry, cells, supports, answers, targets,
                                  slope, bandwidth=1., chunk_size=4).encoding[0]
        predictions.append(float(((encoded-q) @ b / b.square().sum() + 1) / 2))
    c = 1 + (a*a / (1+a*a)) / math.sqrt(a*a / 128 + cfg.norm_eps)
    expected_predictions = [1/(1+math.exp(-2*a*a)), 1/(1+math.exp(-2*c*c*a*a))]
    assert max(abs(x-y) for x,y in zip(predictions,expected_predictions)) < 1e-12
    assert predictions[1] - predictions[0] > .1
    depth_counterexamples = {
        'omab_once_scalar': float(once[0,0]),
        'omab_twice_scalar': float(twice[0,0]),
        'omab_formula_max_error': err(once,expected),
        'unit_scale': c,
        'readout_before_after': predictions,
        'readout_formula_before_after': expected_predictions,
        'scope': 'legal carrier/Unit/readout inputs; no claim these are a trained checkpoint trajectory',
        'passed': True,
    }

source_files = [
    "src/tabu_lab/models/restoration/backbone.py",
    "src/tabu_lab/models/restoration/readout.py",
    "src/tabu_lab/models/restoration_v53/model.py",
    "src/tabu_lab/models/restoration_v53/readout.py",
    "src/tabu_lab/models/restoration_v53/encoding.py",
    "src/tabu_lab/models/restoration_v54/config.py",
]
output = {
    'scope': 'toy current V54Model, deterministic CPU FP64 forward only; no training or checkpoint validation',
    'utc': datetime.now(timezone.utc).isoformat(),
    'source_sha256': {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in source_files},
    'torch': torch.__version__, 'threads': torch.get_num_threads(),
    'elapsed_seconds': time.perf_counter() - start,
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'records': records,
    'depth_counterexamples': depth_counterexamples,
    'passed': all(r['passed'] for r in records),
}
with args.output.open('x') as f:
    f.write(json.dumps(output, indent=2) + '\n')
print(json.dumps({'output': str(args.output), 'passed': output['passed'], 'cases': len(records),
                  'maximum_decoded_error': max(r['decoded_numeric_max_error'] for r in records)}))
assert output['passed']
