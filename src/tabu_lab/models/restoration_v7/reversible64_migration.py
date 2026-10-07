"""Explicit weights-only transfer; only phi_lift donors have matching semantics."""
import hashlib
import io
from dataclasses import replace
from pathlib import Path
import torch
from .config import V7Config, RowReversible64Config
from .model import V7Model
from .runner import CHECKPOINT_SCHEMA, manifest_digest


def row_reversible64_from_checkpoint(path, *, device='cpu', dtype=torch.float64,
                                     seed=0, row_reversible64=None, expected_sha256=None):
    payload = Path(path).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError('donor SHA-256 mismatch')
    state = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    if state.get('schema') != CHECKPOINT_SCHEMA or state.get('architecture') != 'restoration_v7':
        raise ValueError('unsupported donor schema')
    config = V7Config.from_dict(state['config'])
    if config.value_encoder != 'phi_lift':
        raise ValueError('donor must use phi_lift; 128-wide dual-stream semantics differ')
    if state.get('model_version', config.model_version) != config.model_version or state.get('share_rounds', config.share_rounds) != config.share_rounds:
        raise ValueError('donor identity mismatch')
    if state.get('manifest_sha256') != manifest_digest(state['manifest']):
        raise ValueError('donor manifest digest mismatch')
    weights = state['model']
    dtypes = {v.dtype for v in weights.values() if v.is_floating_point()}
    if len(dtypes) != 1 or any(not bool(torch.isfinite(v).all()) for v in weights.values()):
        raise ValueError('nonfinite or mixed-dtype donor')
    with torch.random.fork_rng(devices=[]):
        donor = V7Model(config).to(dtype=next(iter(dtypes)))
        donor.load_state_dict(weights, strict=True)
        cfg = row_reversible64 or RowReversible64Config(
            tau_presence=config.backbone.tau_presence,
            reference_mass=config.backbone.reference_mass, norm_eps=config.backbone.norm_eps)
        torch.manual_seed(seed)
        model = V7Model(replace(config, value_encoder='row_reversible64', row_reversible64=cfg))
    model.to(device=device, dtype=dtype)
    fresh = model.state_dict()
    if set(weights) - set(fresh):
        raise ValueError('unmapped donor keys')
    new = sorted(set(fresh) - set(weights))
    if not new or any('.row_encoder.' not in key for key in new):
        raise ValueError('unclassified new tensor')
    copied = []
    for key, value in weights.items():
        if fresh[key].shape != value.shape:
            raise ValueError('incompatible shape: '+key)
        fresh[key] = value.to(device=device, dtype=dtype)
        copied.append(dict(key=key, shape=list(value.shape)))
    model.load_state_dict(fresh, strict=True)
    for key, value in weights.items():
        if not torch.equal(model.state_dict()[key], value.to(device=device, dtype=dtype)):
            raise ValueError('inexact converted transfer: '+key)
    model.transfer_receipt = dict(
        source_path=str(path), source_sha256=digest, source_step=state['step'],
        donor_config=config.as_dict(), target_config=model.config.as_dict(), seed=seed,
        copied_tensors=copied, copied_tensor_count=len(copied),
        new_tensors=[dict(key=k, shape=list(fresh[k].shape), reason='fresh closed row module') for k in new],
        not_copied=[], new_tensor_count=len(new), load_strict=True,
        optimizer_resume=False, ignored_donor_state=['optimizer','torch_cpu_rng','accelerator_rng','sampler'],
        initialization='zero attention output and final FFN projection; exact identity G',
        target_device=str(device), target_dtype=str(dtype))
    return model
