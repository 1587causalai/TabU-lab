"""Materialize 100 sparse_relevance.1 worlds; no model training."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
TFM=Path('/Users/cms/wehub-system-msp/causal-superintelligence/TFM-Data')
sys.path.insert(0,str(TFM/'src'))
from tfm_data.generators.sparse_relevance import sample_world,sample_rows

def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(obj,separators=(',',':'),allow_nan=False)+'\n').encode()
    with path.open('xb') as f:f.write(raw)
    return hashlib.sha256(raw).hexdigest()

def main():
    records=[]
    for i in range(100):
        world_seed=2026092800+i;row_seed=2026192800+i
        world=sample_world(world_seed=world_seed,teacher='random_gate',n_signal=6)
        values,truth=sample_rows(world,row_seed=row_seed,n_rows=8192,n_features=33,
            nuisance_mode='independent',feature_scale='puma',column_order='permuted',signal_sd=.030,noise_sd=.020)
        name=f'sparse_{i:03d}'
        table=dict(schema='tfm-data.sparse-relevance-table.1',values=values.tolist(),
            features=[dict(key=f'x_{j:02d}',kind='numeric',domain=[]) for j in range(32)]+[dict(key='target',kind='numeric',domain=[])],
            splits=dict(train=list(range(6553)),test=list(range(6553,8192))))
        dh=write(ROOT/'data'/f'{name}.json',table);th=write(ROOT/'truth'/f'{name}.json',truth)
        records.append(dict(id=name,path=f'data/{name}.json',sha256=dh,truth_path=f'truth/{name}.json',truth_sha256=th,world_seed=world_seed,row_seed=row_seed,target_column=32))
    receipt=dict(schema='tabu.sparse-relevance100.data.v1',family='sparse_relevance.1',status='data_prepared_no_training',
        teacher='random_gate',tables=records,n_tables=100,rows_per_table=8192,train_rows=6553,test_rows=1639,
        n_signal=6,n_independent_nuisance=26,signal_sd=.030,noise_sd=.020,column_order='permuted',
        generator=str(TFM/'src/tfm_data/generators/sparse_relevance.py'),generator_sha256=hashlib.sha256((TFM/'src/tfm_data/generators/sparse_relevance.py').read_bytes()).hexdigest(),
        note='Each table has an independent random-gate world. Test rows share its training world; this is not unseen-world generalization. Oracle truth is separate and is not model input.')
    write(ROOT/'data-manifest.json',receipt)
    print(json.dumps(dict(status=receipt['status'],tables=len(records),total_rows=819200),ensure_ascii=False))
if __name__=='__main__':main()
