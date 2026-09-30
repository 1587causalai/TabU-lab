"""Freeze one 8-level ordinal latent-threshold SCM family and prior replay."""
from pathlib import Path
import collections, copy, hashlib, json, shutil, sys, time
import numpy as np

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent / 'nominal100-20260926'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def seed(*parts):
    text = '/'.join(map(str, ('ordinal8-latent100-v1', *parts)))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'little')


def main():
    started = time.monotonic()
    shutil.copytree(PARENT/'source', ROOT/'source', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(PARENT/'generator-source', ROOT/'generator-source', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    sys.path.insert(0, str(ROOT/'generator-source/src'))
    from tfm_data.generators.mixed_scm import sample_world, sample_rows, canonical_hash
    from tfm_data.corpus.adapt import adapt_table

    thresholds = [float(x) for x in np.linspace(-.9, .9, 7)]
    recipe = {
        'family': 'ordinal8_tanh_latent_threshold_scm_v1',
        'generator': 'tfm_data.generators.mixed_scm.sample_world/sample_rows',
        'worlds': 100, 'rows': 256, 'train_rows': 204, 'reserved_rows': 52,
        'columns': [9, 13, 17, 21],
        'target': '8 ordered levels from one additive latent score and seven fixed increasing thresholds',
        'target_order': list(range(8)), 'thresholds': thresholds,
        'latent_noise': 'independent Gaussian(0,0.5) event noise',
        'inputs': 'numeric sparse tanh additive-noise SCM; root sd1, nonroot sd0.1',
        'target_parents': [2, 3], 'max_parents': 3,
        'world_conditioning': 'Use existing ordinal sampler; accept n_classes==8 and 2 or 3 target parents. No model scores.',
        'row_acceptance': 'At least 4 occurrences of every ordered target level in the 204 train rows; reject independent world/event seeds otherwise. No model-performance filtering.',
        'scope': 'one fixed mechanism per table, 100 independent accepted worlds; true ordinal domain, not unordered labels or post-hoc binning',
    }
    write(ROOT/'recipe.json', recipe)
    entries, worlds = [], []
    rejected = collections.Counter()
    for index in range(100):
        name = f'ordinal_latent_train_{index:03d}'
        width = recipe['columns'][index % len(recipe['columns'])]
        for attempt in range(50000):
            world_seed = seed(index, 'world', attempt)
            event_seed = seed(index, 'events', attempt)
            split_seed = seed(index, 'split')
            world = sample_world(np.random.default_rng(world_seed), n_features=width,
                                 n_rows=256, sigma=.5, type_weights={'numeric': 1.0},
                                 target_type='ordinal', max_parents=3)
            target = world['nodes'][world['target_node']]
            if target['n_classes'] != 8 or len(target['parents']) not in (2, 3):
                rejected['structural'] += 1
                continue
            upstream_hash = canonical_hash(world)
            for node in world['nodes']:
                if node['type'] == 'numeric':
                    node['mechanism']['noise_scale'] = .1 if node['parents'] else 1.0
                for edge in node['mechanism']['edges']:
                    edge['transform'] = 'tanh'
            target['mechanism']['noise_scale'] = .5
            if target['mechanism']['thresholds'] != thresholds:
                raise RuntimeError('existing ordinal generator threshold contract drift')
            values = sample_rows(world, n_rows=256, rng=np.random.default_rng(event_seed))
            episode = {'table': {'values': [[float(x) for x in row[:-1]]+[int(row[-1])] for row in values],
                                 'column_types': ['numeric']*(width-1)+['ordinal'],
                                 'n_classes': [None]*(width-1)+[8]}}
            data = adapt_table(episode, width-1, name=name, split_seed=split_seed)
            train_labels = values[data['splits']['train'], -1].astype(int)
            counts = np.bincount(train_labels, minlength=8)
            if min(counts) < 4:
                rejected['train_level_coverage'] += 1
                continue
            break
        else:
            raise RuntimeError(f'acceptance exhausted for {name}')
        data_path = ROOT/'data/ordinal100'/f'{name}.json'
        world_path = ROOT/'worlds'/f'{name}.json'
        write(data_path, data)
        write(world_path, {'world': world, 'world_seed': world_seed, 'event_seed': event_seed,
                           'split_seed': split_seed, 'attempt': attempt,
                           'upstream_world_hash': upstream_hash})
        worlds.append({'id': name, 'world_sha256': canonical_hash(world),
                       'world_file_sha256': sha(world_path), 'world_path': str(world_path.relative_to(ROOT)),
                       'data_sha256': sha(data_path), 'train_level_counts': counts.tolist(),
                       'accepted_attempt': attempt})
        entries.append({'id': name, 'path': '../'+str(data_path.relative_to(ROOT)),
                        'sha256': sha(data_path), 'cohort': 'ordinal100',
                        'kind': 'synthetic', 'role': 'train'})
        print(name, counts.tolist(), flush=True)

    parent_manifest = json.loads((PARENT/'manifests/candidate.json').read_text())
    for entry in parent_manifest['tables']:
        source = (PARENT/'manifests'/entry['path']).resolve()
        destination = ROOT/'data'/entry['cohort']/source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        assert sha(destination) == entry['sha256']
        entries.append(dict(entry, path='../'+str(destination.relative_to(ROOT))))

    manifest = copy.deepcopy(parent_manifest)
    manifest.update(experiment_id='v55-ordinal100-continuous-learning-dgx2-20260926',
                    description='100 independent 8-level ordinal latent-threshold SCM worlds, 80.0% updates; numeric100, nominal100, old120 and real78 replay. Fresh weights-only stage from nominal u13292.')
    manifest['tables'] = entries
    mask = {'kind': 'supervised_row', 'fraction': .25}
    manifest['probes'] = [
        {'name': name, 'cohorts': [cohort], 'partition': 'train', 'purpose': 'fit',
         'recipe': mask, 'masks': masks}
        for name, cohort, masks in (
            ('ordinal100_fit', 'ordinal100', 4), ('nominal100_fit', 'nominal100', 2),
            ('tanh100_fit', 'tanh100', 2), ('train_fit', 'old120', 1),
            ('real78_fit', 'real78', 1))]
    stage = manifest['stages'][0]
    stage.update(name='ordinal100_with_398_replay', question='Fit ordered targets while retaining numeric, nominal, old120 and real78',
                 max_updates=60000, max_seconds=14400.0,
                 sampling=[{'cohort': cohort, 'episodes': episodes} for cohort, episodes in (
                     ('ordinal100', 1900), ('tanh100', 100), ('nominal100', 100),
                     ('old120', 120), ('real78', 156))],
                 checkpoint_every=2376, evaluate_every=60000,
                 probes=[probe['name'] for probe in manifest['probes']])
    write(ROOT/'manifests/candidate.json', manifest)
    write(ROOT/'corpus-manifest.json', {
        'recipe_sha256': sha(ROOT/'recipe.json'), 'worlds': worlds,
        'rejection_counts': dict(rejected),
        'generator_source_sha256': sha(ROOT/'generator-source/src/tfm_data/generators/mixed_scm.py'),
        'generation_seconds': time.monotonic()-started,
    })
    print('PREPARED', len(entries), dict(rejected), time.monotonic()-started, flush=True)


if __name__ == '__main__':
    main()
