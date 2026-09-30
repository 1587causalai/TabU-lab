#!/usr/bin/env python3
"""CPU-only validation and fixed bank construction for ordinal100."""
from pathlib import Path
import collections, hashlib, json, platform, resource, sys, time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PARENT = ROOT.parent/'nominal100-20260926'
MANIFEST = ROOT/'manifests/candidate.json'
RECIPE = ROOT/'recipe.json'
CORPUS = ROOT/'corpus-manifest.json'
SOURCE = ROOT/'source/src'
GENERATOR = ROOT/'generator-source/src'
sys.path[:0] = [str(SOURCE), str(GENERATOR)]

from tfm_data.generators.mixed_scm import canonical_hash, sample_rows
from tabu_lab.curriculum_v53.data import build_episode
from tabu_lab.curriculum_v53.protocol import load_v55_plan


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode()


def digest(value): return hashlib.sha256(canonical(value)).hexdigest()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(condition, message):
    if not condition: raise AssertionError(message)


def main():
    started = time.monotonic()
    manifest = json.loads(MANIFEST.read_text())
    recipe = json.loads(RECIPE.read_text())
    corpus = json.loads(CORPUS.read_text())
    parent_manifest = json.loads((PARENT/'manifests/candidate.json').read_text())
    parent_plan = json.loads((PARENT/'evidence/validation/plan.json').read_text())
    plan = load_v55_plan(MANIFEST)
    counts = collections.Counter(table.cohort for table in plan.tables)
    expected = collections.Counter(ordinal100=100, nominal100=100, tanh100=100,
                                   old120=120, real78=78)
    require(counts == expected, f'cohort counts {counts}')
    require(len(plan.tables) == 498, 'table count')
    sampling = [(item['cohort'], item['episodes']) for item in manifest['stages'][0]['sampling']]
    require(sampling == [('ordinal100',1900),('tanh100',100),('nominal100',100),
                         ('old120',120),('real78',156)], f'sampling {sampling}')
    require(manifest['model'] == parent_manifest['model'], 'model config drift')
    require(manifest['optimizer'] == parent_manifest['optimizer'], 'optimizer config drift')
    expected_files = parent_plan['identity']['source']['files']
    source_drift = [rel for rel, expected_sha in expected_files.items()
                    if sha(SOURCE/'tabu_lab'/rel) != expected_sha]
    require(not source_drift, f'source drift {source_drift[:5]}')

    parent_entries = {entry['id']:entry for entry in parent_manifest['tables']}
    current_entries = {entry['id']:entry for entry in manifest['tables']}
    require(set(parent_entries) < set(current_entries), 'parent table set not preserved')
    for name, old in parent_entries.items():
        new = current_entries[name]
        require(new['sha256'] == old['sha256'] and new['cohort'] == old['cohort'],
                f'replay identity drift {name}')
        require(sha((MANIFEST.parent/new['path']).resolve()) == old['sha256'],
                f'replay bytes drift {name}')

    world_hashes, upstream_hashes, data_hashes, seeds = set(), set(), set(), set()
    parent_counts, widths = collections.Counter(), collections.Counter()
    thresholds = recipe['thresholds']
    for record in corpus['worlds']:
        name = record['id']
        world_file_path = ROOT/record['world_path']
        data_path = ROOT/'data/ordinal100'/f'{name}.json'
        world_file = json.loads(world_file_path.read_text())
        data = json.loads(data_path.read_text())
        world = world_file['world']
        require(sha(world_file_path) == record['world_file_sha256'], f'world bytes {name}')
        require(sha(data_path) == record['data_sha256'], f'data bytes {name}')
        require(canonical_hash(world) == record['world_sha256'], f'world hash {name}')
        world_hashes.add(record['world_sha256']); upstream_hashes.add(world_file['upstream_world_hash'])
        data_hashes.add(record['data_sha256']); seeds.add((world_file['world_seed'],world_file['event_seed']))
        widths[world['n_features']] += 1
        target = world['nodes'][world['target_node']]
        parent_counts[len(target['parents'])] += 1
        require(target['type']=='ordinal' and target['n_classes']==8, f'target type {name}')
        require(len(target['parents']) in (2,3), f'target parents {name}')
        mechanism = target['mechanism']
        require(mechanism['family']=='additive_score' and mechanism['noise_family']=='gaussian'
                and mechanism['noise_scale']==.5, f'latent mechanism {name}')
        require(mechanism['thresholds']==thresholds and all(a<b for a,b in zip(thresholds,thresholds[1:])),
                f'threshold contract {name}')
        for node in world['nodes']:
            if node['node'] != world['target_node']:
                require(node['type']=='numeric', f'nonnumeric input {name}')
            if node['type']=='numeric':
                require(node['mechanism']['noise_scale']==(1.0 if not node['parents'] else .1),
                        f'numeric noise {name}')
            require(all(edge['transform']=='tanh' for edge in node['mechanism']['edges']),
                    f'edge transform {name}')
        values = sample_rows(world, n_rows=256, rng=np.random.default_rng(world_file['event_seed']))
        require(all(float(a)==float(b) for row_a,row_b in zip(values.tolist(),data['values'])
                    for a,b in zip(row_a,row_b)), f'row replay {name}')
        require(data['target_kind']=='ordinal' and data['features'][-1]['kind']=='ordinal',
                f'ordinal data contract {name}')
        require(data['domain']==[str(i) for i in range(8)]
                and data['features'][-1]['domain']==data['domain'], f'order domain {name}')
        require(all(feature['kind']=='numeric' for feature in data['features'][:-1]), f'inputs {name}')
        train = data['splits']['train']
        level_counts = np.bincount([int(data['values'][row][-1]) for row in train], minlength=8)
        require(min(level_counts)>=4, f'train level coverage {name}: {level_counts.tolist()}')
    require(len(corpus['worlds'])==len(world_hashes)==len(upstream_hashes)==len(data_hashes)==len(seeds)==100,
            'world independence')
    require(dict(widths)=={9:25,13:25,17:25,21:25}, f'widths {widths}')

    probes = {probe['cohorts'][0]:probe for probe in manifest['probes']}
    bank_entries, ordinal_masks = [], []
    for table in plan.tables:
        probe = probes[table.cohort]
        masks = []
        for index in range(probe['masks']):
            _,_,_,info = build_episode(table,probe['recipe'],index,plan.spec['seeds'],'cpu',
                                       codec_version=manifest['model']['codec_version'])
            query_rows = sorted(info['query_row_ids'])
            support_rows = sorted(set(info['row_ids'])-set(query_rows))
            row_to_index = {int(row):position for position,row in enumerate(table.row_ids)}
            labels = [int(table.values[table.target_column][row_to_index[row]]) for row in support_rows]
            row = {'index':info['index'],'mask_seed':info['mask_seed'],'code_seed':info['code_seed'],
                   'window_seed':info['window_seed'],'target_column':table.target_column,
                   'query_count':info['query_count'],'support_count':len(support_rows),
                   'query_addresses':info['query_addresses'],'query_rows':query_rows,
                   'support_rows':support_rows,'support_target_classes':sorted(set(labels)),
                   'row_ids':info['row_ids']}
            require(row['query_count']==51 and row['support_count']==153, f'mask size {table.name}/{index}')
            if table.cohort=='ordinal100':
                require(row['support_target_classes']==list(range(8)),
                        f'missing ordinal support level {table.name}/{index}: {row["support_target_classes"]}')
                require(all(address[1]==table.target_column for address in row['query_addresses']),
                        f'query target {table.name}/{index}')
                ordinal_masks.append({'table':table.name,**row})
            masks.append(row)
        bank_entries.append([table.name,probe['name'],masks])
    bank_sha = digest(bank_entries)
    bank = {'schema':'tabu.ordinal100.fixed-bank.v1',
            'contract':'train bank: 204 rows, 153 support + 51 Query target cells',
            'table_count':len(bank_entries),'tables':bank_entries,'bank_sha256':bank_sha,
            'validation_tables':[],'validation_bank_sha256':None}
    (OUT/'fixed-bank-addresses.json').write_text(json.dumps(bank,sort_keys=True,indent=2)+'\n')
    plan_receipt = {'schema':'tabu.ordinal100.candidate-plan-receipt.v1','manifest':str(MANIFEST),
                    'manifest_sha256':sha(MANIFEST),'identity':plan.identity,'source':plan.identity['source'],
                    'spec':plan.spec,'summary':plan.summary,'model':manifest['model'],'optimizer':manifest['optimizer'],
                    'parent_checkpoint':{'update':13292,'sha256':'cb0ed2b5e80612de00a1888dc8a345132f36dc92a3fa097ea3a473065135a776'},
                    'initialization':'weights_only; fresh AdamW/RNG/cursor/exposure because the data plan changed',
                    'source_check':{'candidate_source_sha256':plan.identity['source']['sha256'],
                                    'same_bytes':len(expected_files),'different_files':source_drift},
                    'model_same_as_parent':True,'optimizer_config_same_as_parent':True,
                    'outcome':'passed','execution_started':False}
    (OUT/'plan.json').write_text(json.dumps(plan_receipt,sort_keys=True,indent=2)+'\n')
    episode = {'schema':'tabu.ordinal100.candidate-episode-validation.v1','outcome':'passed',
               'manifest_sha256':sha(MANIFEST),'identity_sha256':plan.identity['sha256'],
               'source_sha256':plan.identity['source']['sha256'],'cohorts':dict(counts),
               'sampling':sampling,'new_fraction':1900/2376,'replay_fraction':476/2376,
               'world_replay':{'worlds':100,'unique_world_hashes':len(world_hashes),
                   'unique_upstream_hashes':len(upstream_hashes),'widths':dict(widths),
                   'target_parent_counts':dict(parent_counts),'rejection_counts':corpus['rejection_counts'],
                   'generator_source_sha256':corpus['generator_source_sha256'],
                   'recipe_sha256':sha(RECIPE),'generation_seconds':corpus['generation_seconds']},
               'ordinal100_bank':{'tables_checked':100,'masks_checked':len(ordinal_masks),
                                  'query_count':51,'support_count':153,'ordered_levels':list(range(8))},
               'ordinal100_mask_receipts':ordinal_masks,'fixed_bank_sha256':bank_sha,
               'partition_isolation':{'train_rows':204,'reserved_rows':52,'validation_present':False},
               'runtime':{'python':sys.version,'platform':platform.platform(),
                          'peak_rss_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)},
               'elapsed_seconds':time.monotonic()-started,'execution_started':False}
    (OUT/'candidate-episode-validation.json').write_text(json.dumps(episode,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'outcome':'passed','identity_sha256':plan.identity['sha256'],
                      'source_sha256':plan.identity['source']['sha256'],'manifest_sha256':sha(MANIFEST),
                      'bank_sha256':bank_sha,'table_count':len(bank_entries),
                      'ordinal_masks':len(ordinal_masks),'elapsed_seconds':episode['elapsed_seconds']}))


if __name__=='__main__': main()
