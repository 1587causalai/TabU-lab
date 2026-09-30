"""Isolated pre-projection random column mixing; legacy V6 is untouched."""
import torch

from tabu_lab.models.restoration._validation import finite
from tabu_lab.models.restoration.model import ColumnPrediction
from tabu_lab.models.restoration.readout import EncodedRestoration
from tabu_lab.models.restoration_v53.encoding import ANSWER_WIDTH
from tabu_lab.models.restoration_v53.model import PreparedV53, V53Output
from tabu_lab.models.restoration_v53.readout import evaluate_column, shared_slope
from tabu_lab.models.restoration_v55 import V55Config, V55Model


import hashlib, json, random
from torch.nn import functional as F
from tabu_lab.models.restoration_v6 import V6Model
from tabu_lab.models.restoration_v6.model import task_target_column

MIX_CONFIG = dict(kind="random_two_nontarget_columns", donors=2, coefficient=1.0,
    location="raw_cell_encoding_before_projection", target_broadcast="original_unmixed_target_carrier",
    mapping="episode_code_seed_and_stable_column_keys", same_mapping_all_rows=True,
    destination="all_active_cells_including_target", source="visible_non_target_cells_only",
    shortfall="use_available_distinct_donors", namespace="tabu-randommix2-v1")

def donor_map(inputs):
    target = task_target_column(inputs)
    keys = [s.key for s in inputs.schema]
    choices = []
    for j,key in enumerate(keys):
        eligible = sorted((k for k in range(len(keys)) if k not in (target,j)), key=lambda k:keys[k])
        seed = int.from_bytes(hashlib.sha256(json.dumps([MIX_CONFIG['namespace'],inputs.code_seed,key,keys[target]]).encode()).digest()[:8],'little')
        choices.append(random.Random(seed).sample(eligible,min(2,len(eligible))))
    return choices

def encode_pair(encoder, inputs, layout, *, enabled=True):
    # Query truth is absent: layout contains visible encodings only.
    n,m=inputs.visible.shape;weight=encoder.projection.weight
    raw=weight.new_zeros(((n+1)*(m+1),ANSWER_WIDTH))
    lifts=[]
    for kind,coordinates,fixed_rank in layout.groups:
        lift=coordinates.to(weight)
        if kind=='ordinal' and encoder.codec_version=='legacy_v53':
            lift=lift+fixed_rank.to(weight)[:,None]
        lifts.append(lift)
    raw=raw.index_copy(0,layout.addresses,torch.cat(lifts)).reshape(n+1,m+1,ANSWER_WIDTH)
    raw_cells=raw[:n,:m]
    assert not bool(raw_cells[~inputs.visible].any())
    choices=donor_map(inputs)
    extra=torch.stack([sum((raw_cells[:,k] for k in donors),torch.zeros_like(raw_cells[:,j])) for j,donors in enumerate(choices)],dim=1)
    # Add in encoding space, before the shared bias-free linear projection.
    base=weight.new_zeros((n+1,m+1,encoder.width))
    base[:n,:m]=F.linear(raw_cells,weight)+torch.where(inputs.query[...,None],encoder.cell_seed,0)
    base[:n,m]=encoder.unit_seed;base[n,:m]=encoder.feature_seed
    mixed=base.clone()
    if enabled:
        mixed[:n,:m]=torch.where((inputs.visible|inputs.query)[...,None],
            F.linear(raw_cells+extra,weight)+torch.where(inputs.query[...,None],encoder.cell_seed,0),0)
    finite(mixed,'random two-column mixed carriers')
    mapping=dict(config=MIX_CONFIG,donors=choices,target_column=task_target_column(inputs),
        sha256=hashlib.sha256(json.dumps(choices).encode()).hexdigest(),code_seed=inputs.code_seed)
    return base,mixed,mapping

class MixedV6Model(V6Model):
    def __init__(self,config,*,supervision='target_only'):
        super().__init__(config,supervision=supervision)
        if supervision!='target_only':raise ValueError('This experiment freezes target_only loss')
        self.mix_enabled=True
        self.last_mix=None
    def forward_prepared(self, prepared: PreparedV53, *, decode: bool = True) -> V53Output:
        if prepared.encoder_id != id(self.encoder) or prepared.config != self.config:
            raise ValueError("prepared V6 model changed")
        prepared.versions.validate()
        inputs, request, facts = prepared.inputs, prepared.request, prepared.facts
        target = task_target_column(inputs)
        n, m = inputs.visible.shape

        base, mixed, mapping = encode_pair(self.encoder, inputs, prepared.features, enabled=self.mix_enabled)
        self.last_mix = mapping
        active = inputs.visible | inputs.query
        cells = torch.where(
            active[..., None],
            mixed[:n, :m] + base[:n, target][:, None, :],
            torch.zeros_like(base[:n, :m]),
        )
        h = torch.cat((torch.cat((cells, base[:n, m:m + 1]), dim=1), base[n:]), dim=0)
        finite(h, "V6 broadcast carriers")
        h = self.backbone(h, inputs.visible, inputs.query)
        units = h[:n, m]
        eligible = inputs.visible.any(-1)
        for block in self.unit_blocks:
            units = block(units, units, eligible)

        # Only the optional joint response needs target codes from visible
        # support labels. Query truth is never passed to this forward map.
        target_fact = facts[target]
        target_codes = None
        if self.supervision == "joint_all":
            target_codes = target_fact.answers.encoded.new_zeros(n, ANSWER_WIDTH)
            target_codes = target_codes.index_copy(0, target_fact.rows,
                                                   target_fact.answers.encoded)
        columns, slopes = [], {}
        for a, positions in enumerate(prepared.positions):
            if not len(positions):
                continue
            fact = facts[a]
            if self.supervision == "joint_all":
                complete = inputs.visible[fact.rows, target]
                support_rows = fact.rows[complete]
                answers = fact.answers.encoded[complete] + target_codes[support_rows]
            else:
                support_rows = fact.rows
                answers = fact.answers.encoded
            if not len(support_rows):
                columns.append(ColumnPrediction(
                    a, positions, EncodedRestoration("no-support", 0), None
                ))
                continue
            finite(answers, "V6 support answers")
            regression_cells = self.regression(h[:n, a])
            finite(regression_cells, "V6 regression Cell features")
            slope = shared_slope(
                units, support_rows, regression_cells[support_rows], answers,
                ridge=self.config.ridge, bandwidth=self.config.bandwidth,
                center_chunk_size=self.config.center_chunk_size,
            )
            slopes[a] = slope
            result = evaluate_column(
                units, regression_cells, support_rows, answers,
                request.targets[positions, 0], slope,
                bandwidth=self.config.bandwidth,
                chunk_size=self.config.center_chunk_size,
            )
            decoded = None
            if decode:
                if self.supervision == "target_only":
                    decoded = fact.answers.decode(result.encoding)
                elif a == target:
                    decoded = target_fact.answers.decode(result.encoding / 2)
            columns.append(ColumnPrediction(a, positions, result, decoded))
        return V53Output(request, tuple(columns), facts, h, units, slopes)
