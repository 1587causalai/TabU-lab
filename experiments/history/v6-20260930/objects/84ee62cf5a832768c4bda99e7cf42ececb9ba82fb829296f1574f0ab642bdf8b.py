"""Verify unchanged schedule/RNG continuation and four disposable branch updates."""
import copy
import json
import sys
from collections import Counter
from pathlib import Path

B = Path(__file__).resolve().parent
sys.path.insert(0, str(B))
import torch
from train import rewrite_spec, validate_plan, OBJECTIVE
from dual import DualModel, prepare, score, V53LossConfig
from branch_sampling import BranchSampler
from training_support import episode_seeds, restore_rng
from tabu_lab.curriculum_v53.protocol import load_v55_plan, schedule_entry
from tabu_lab.curriculum_v53.artifacts import sha256, atomic_json, finite_state
from tabu_lab.curriculum_v53.data import build_episode
from tabu_lab.curriculum_v53.runner import configure_runtime
from tabu_lab.models.restoration._dtype import execution_dtype
from tabu_lab.restoration_optimizers import adamw


class CountModel(DualModel):
    def forward_prepared(self, *args, **kwargs):
        self.calls += 1
        return super().forward_prepared(*args, **kwargs)


def main():
    q = json.loads((B/'queue.json').read_text())
    assert q['host'] == 'gongqian-mini' and q['device'] == 'mps'
    parent = q['parent']
    runtime = configure_runtime('mps')
    torch.mps.set_per_process_memory_fraction(.75)
    assert sha256(parent['checkpoint']) == parent['checkpoint_sha256']
    donor = torch.load(parent['checkpoint'], map_location='cpu', weights_only=False)
    old = load_v55_plan(parent['manifest'])
    assert donor['identity']['parent_manifest_sha256'] == sha256(parent['manifest'])
    assert donor['identity']['objective'] == OBJECTIVE
    spec = rewrite_spec(old.spec, Path(parent['manifest']).parent, B, q['host'])
    atomic_json(B/'preflight-manifest.json', spec)
    plan = load_v55_plan(B/'preflight-manifest.json')
    validate_plan(plan)
    assert plan.config.as_dict() == old.config.as_dict() and plan.optimizer == old.optimizer
    assert plan.spec['seeds'] == old.spec['seeds']
    assert plan.spec['stages'] == old.spec['stages']
    # Only experiment identity and equivalent relative file paths may differ.
    for a, b in zip(old.spec['tables'], plan.spec['tables'], strict=True):
        assert {k:v for k,v in a.items() if k!='path'} == {k:v for k,v in b.items() if k!='path'}
        assert (Path(parent['manifest']).parent/a['path']).resolve() == (B/b['path']).resolve()
    cursor = donor['state']['cursor']
    offsets = copy.deepcopy(donor['state']['table_episode_offsets'])
    assert len(offsets) == 730
    counts = Counter()
    for i in range(cursor, cursor+2700):
        t, ep = schedule_entry(plan, 0, i)
        ot, oe = schedule_entry(old, 0, i)
        assert (t.name, ep) == (ot.name, oe)
        counts[t.cohort] += 1
    for cycle in range(cursor//900, cursor//900+3):
        cs = Counter(schedule_entry(plan, 0, i)[0].cohort for i in range(cycle*900, (cycle+1)*900))
        assert cs == Counter(sparse100=270, history718=618, focus2=2, other10=10)
    selector = BranchSampler(donor['state']['branch_rng_state'])
    again = BranchSampler(donor['state']['branch_rng_state'])
    assert [selector.draw() for _ in range(200)] == [again.draw() for _ in range(200)]
    saved = selector.state()
    expected = [selector.draw() for _ in range(100)]
    restored = BranchSampler(saved)
    assert expected == [restored.draw() for _ in range(100)]
    model = CountModel(plan.config).to('mps', dtype=execution_dtype('mps'))
    model.load_state_dict(donor['model'], strict=True)
    opt = adamw(model, plan.optimizer)
    opt.load_state_dict(donor['optimizer'])
    restore_rng(donor['rng'])
    config = V53LossConfig(**plan.spec['stages'][0]['loss'])
    seeds = episode_seeds(plan)
    tests = []
    for name in ['sparse_000', 'pumadyn32nh']:
        pos = next(i for i in range(cursor, cursor+1800) if schedule_entry(plan, 0, i)[0].name == name)
        table, ep = schedule_entry(plan, 0, pos)
        inputs, request, truth, _ = build_episode(table, plan.spec['stages'][0]['recipe'][table.kind], offsets[name]+ep, seeds, 'mps', epsilon=plan.config.epsilon, codec_version=plan.config.codec_version)
        for branch in ['v6', 'v55']:
            model.calls = 0
            opt.zero_grad(set_to_none=True)
            prepared = prepare(model, inputs, request, truth, config)
            result = score(model, prepared, config, branch)
            assert model.calls == 1 and len(prepared.visible.request.targets) == int(inputs.query.sum())
            result.loss.backward()
            grads = [p for p in model.parameters() if p.grad is not None]
            assert grads and finite_state([p.grad for p in grads])
            torch.nn.utils.clip_grad_norm_(grads, plan.optimizer.grad_clip, error_if_nonfinite=True)
            opt.step()
            assert finite_state(model.state_dict()) and finite_state(opt.state_dict())
            torch.mps.synchronize()
            tests.append(dict(table=name, branch=branch, rows=inputs.query.shape[0], query_cells=int(inputs.query.sum()), forward_passes=model.calls, optimizer_steps=1, loss=float(result.loss.detach())))
            del result, prepared
    previous = B.parent/parent['source_run']
    unchanged = ['dual.py', 'branch_sampling.py', 'training_support.py', 'evaluate.py', 'eval_fit.py']
    assert all(sha256(B/f) == sha256(previous/f) for f in unchanged)
    assert sha256(parent['checkpoint']) == parent['checkpoint_sha256']
    report = dict(outcome='passed', runtime=runtime, parent_sha256=parent['checkpoint_sha256'], parent_unchanged=True, disposable_updates=4, tests=tests, parent_cursor=cursor, schedule_cursor_reset=False, next_2700_schedule_and_episodes_equal=True, table_offsets_preserved=True, branch_rng_restored=True, branch_rng_roundtrip=True, unchanged_code_hashes={f:sha256(B/f) for f in unchanged})
    atomic_json(B/'preflight.json', report)
    print(json.dumps({k:report[k] for k in ['outcome','disposable_updates','parent_cursor','schedule_cursor_reset','next_2700_schedule_and_episodes_equal','branch_rng_restored']}))


if __name__ == '__main__':
    main()
