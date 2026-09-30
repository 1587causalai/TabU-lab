"""Isolated residual encoder; original V6 and readout source are unchanged."""
import torch
from torch import nn
from tabu_lab.models.restoration_v53.encoding import AffineValueEncoder
from tabu_lab.models.restoration_v6 import V6Model

class ResidualEncoder(AffineValueEncoder):
    def __init__(self, config):
        super().__init__(config.backbone.width, config.epsilon,
            codec_version=config.codec_version, numeric_scaling=config.numeric_scaling)
        self.residual_v = nn.Linear(self.width, self.width, bias=False)
        self.residual_u = nn.Linear(self.width, self.width, bias=False)
        nn.init.zeros_(self.residual_u.weight)

    def forward_prepared(self, inputs, layout):
        h = super().forward_prepared(inputs, layout)
        # layout.addresses contains visible values only; all semantic seeds and
        # Null slots retain exactly the old compiler path.
        flat = h.flatten(0, 1)
        visible = flat.index_select(0, layout.addresses)
        delta = self.residual_u(torch.nn.functional.gelu(self.residual_v(visible)))
        return flat.index_copy(0, layout.addresses, visible + delta).reshape_as(h)

def make_model(config, donor, arm, seed):
    torch.manual_seed(seed)
    model = V6Model(config, supervision='target_only')
    if arm == 'residual_gelu':
        model.encoder = ResidualEncoder(config)
    else:
        assert arm == 'linear'
    missing, unexpected = model.load_state_dict(donor['model'], strict=False)
    expected = ['encoder.residual_v.weight', 'encoder.residual_u.weight'] if arm=='residual_gelu' else []
    assert set(missing)==set(expected) and not unexpected, (missing, unexpected)
    return model.to('mps', dtype=torch.float32)

def make_optimizer(model, cfg, donor):
    decay, other, added = [], [], []
    for name,p in model.named_parameters():
        if name.startswith('encoder.residual_'): added.append(p)
        elif name.endswith('.weight'): decay.append(p)
        else: other.append(p)
    optimizer=torch.optim.AdamW([{'params':decay,'weight_decay':cfg.weight_decay},
        {'params':other,'weight_decay':0.0}],lr=cfg.learning_rate,betas=cfg.betas,eps=cfg.eps)
    optimizer.load_state_dict(donor['optimizer'])
    if added:
        group={k:v for k,v in optimizer.param_groups[0].items() if k!='params'}
        optimizer.add_param_group(dict(group,params=added))
    assert {id(p) for g in optimizer.param_groups for p in g['params']}=={id(p) for p in model.parameters()}
    return optimizer
