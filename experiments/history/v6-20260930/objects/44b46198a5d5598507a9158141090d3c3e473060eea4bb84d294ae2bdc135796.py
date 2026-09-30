"""Fixed 100-table synthetic test bank: 8 x (136 support + 68 query)."""
import argparse,json,math,statistics,sys,time
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from evaluation_support import load_v6_checkpoint,score_log
from frozen_evaluate import make_input,metrics,state_hash
from tabu_lab.curriculum_v53.artifacts import atomic_json,sha256
from tabu_lab.curriculum_v53.protocol import load_v55_plan
from tabu_lab.curriculum_v53.runner import configure_runtime
from tabu_lab.models.restoration._dtype import execution_dtype
from tabu_lab.models.restoration_v6 import V6Model

def main(args):
    base=Path(__file__).resolve().parent;out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    report=dict(outcome='running',checkpoint_sha256=args.expected_sha,bank_sha256=sha256(base/'evaluation-bank.json'),evaluation='100 known worlds; 8 fixed disjoint test-query windows each; 136 train support + 68 test Query',optimizer_updates=0)
    atomic_json(out/'started.json',report)
    try:
        runtime=configure_runtime(args.device);cap=.75 if args.host=='gongqian-mini' else .5
        if args.device=='mps':torch.mps.set_per_process_memory_fraction(cap)
        else:torch.cuda.set_per_process_memory_fraction(cap)
        plan=load_v55_plan(args.manifest);payload=load_v6_checkpoint(args.checkpoint,args.expected_sha)
        assert payload['model_config']==plan.config.as_dict()
        model=V6Model(plan.config,supervision=payload['identity']['supervision']).to(args.device,dtype=execution_dtype(args.device))
        model.load_state_dict(payload['model'],strict=True);model.eval().requires_grad_(False)
        before=state_hash(model);report.update(runtime=runtime,checkpoint_update=payload['state']['update'],supervision=payload['identity']['supervision']);del payload,plan
        bank=json.loads((base/'evaluation-bank.json').read_text())
        for entry in bank['tables']:
            path=base/entry['path'];assert sha256(path)==entry['sha256']
            data=json.loads(path.read_text());rows=[];tick=time.monotonic()
            for episode,trace in enumerate(entry['traces']):
                inputs,request=make_input(data,trace,entry['name'],args.device,execution_dtype(args.device))
                with torch.inference_mode():answer=model(inputs,request)
                assert len(answer.columns)==1
                column=answer.columns[0];assert column.result.status=='ok' and column.column==32 and column.decoded is not None
                preds=column.decoded.detach().cpu().tolist();positions=column.target_indices.detach().cpu().tolist();addresses=request.targets.detach().cpu().tolist();seen=set()
                assert len(preds)==68
                for pred,pos in zip(preds,positions):
                    r,c=addresses[pos];assert c==32 and 136<=r<204 and math.isfinite(pred)
                    row_id=trace['query_row_ids'][r-136];assert row_id not in seen;seen.add(row_id)
                    rows.append(dict(episode=episode,row_id=row_id,target=data['values'][row_id][-1],prediction=pred))
                assert seen==set(trace['query_row_ids'])
                del answer,column,inputs,request
            assert len(rows)==len({r['row_id'] for r in rows})==544
            result=metrics(rows);result['slog']=score_log(rows,[data['values'][i][-1] for i in data['splits']['train']])
            report.setdefault('tables',{})[entry['name']]=dict(**result,seconds=time.monotonic()-tick,support_rows_per_forward=136,query_rows_per_forward=68)
            atomic_json(out/(entry['name']+'-predictions.json'),rows);atomic_json(out/'progress.json',report)
            print(json.dumps(dict(table=entry['name'],r2=result['r2'],slog=result['slog'])),flush=True)
        assert state_hash(model)==before
        report.update(outcome='completed',total_predictions=sum(x['n'] for x in report['tables'].values()),macro={k:statistics.fmean(x[k] for x in report['tables'].values()) for k in ['r2','slog']})
        assert len(report['tables'])==100 and report['total_predictions']==54400
    except Exception as e:
        report.update(outcome='failed',error_type=type(e).__name__,error=str(e));atomic_json(out/'terminal.json',report);raise
    atomic_json(out/'terminal.json',report)
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['host','device','manifest','checkpoint','expected-sha','output']:parser.add_argument('--'+name,required=True)
    main(parser.parse_args())
