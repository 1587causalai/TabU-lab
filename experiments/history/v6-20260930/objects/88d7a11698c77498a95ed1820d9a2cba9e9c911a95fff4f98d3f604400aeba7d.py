"""Predeclared CPU MLP/XGBoost fit-capacity probes, no test-based selection."""
import json,math,time,os,platform,hashlib,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import torch
import xgboost as xgb
from common import B,load_data,metric_rows
from tabu_lab.curriculum_v53.artifacts import atomic_json,append_event,sha256

def run():
    out=B/'baselines';out.mkdir(exist_ok=False)
    data,bank=load_data()
    raw=np.asarray(data['values'],dtype=np.float64)
    train=np.asarray(data['splits']['train']);test=np.asarray(data['splits']['test'])
    x,y=raw[:,:-1],raw[:,-1]
    xm=x[train].mean(0);xs=x[train].std(0);xs[xs<1e-12]=1
    ym=y[train].mean();ys=y[train].std()
    xt=((x[train]-xm)/xs).astype(np.float32);yt=((y[train]-ym)/ys).astype(np.float32)
    xe=((x[test]-xm)/xs).astype(np.float32)
    def scores(pt,pe):
        return {part:metric_rows([dict(row_id=int(i),target=float(y[i]),prediction=float(p)) for i,p in zip(ids,pred,strict=True)],data)
                for part,ids,pred in [('train',train,pt),('test',test,pe)]}
    result=dict(outcome='running',host=platform.node(),device='cpu',dtype='float32',
                train_n=len(train),test_n=len(test),bank_sha256=sha256(B/'fit-bank.json'),
                selection='No test-based tuning. Report every fixed candidate; train-fit goal, no validation early stopping.',
                limits='Each of two XGBoost candidates <=140 fit seconds; each of two MLP candidates <=300 fit seconds. Round/epoch boundary overshoot only.',
                versions={'torch':torch.__version__,'xgboost':xgb.__version__,'numpy':np.__version__},models={})
    atomic_json(out/'status.json',result)
    started=time.monotonic()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    try:
        class Limit(xgb.callback.TrainingCallback):
            def __init__(self):self.tick=None;self.elapsed=0
            def before_training(self,model):self.tick=time.monotonic();return model
            def after_iteration(self,model,epoch,evals_log):
                self.elapsed=time.monotonic()-self.tick
                if epoch%100==0:
                    append_event(out/'xgb-trace.jsonl',dict(depth=depth,trees=epoch+1,seconds=self.elapsed,train_rmse_standardized=evals_log['train']['rmse'][-1]))
                return self.elapsed>=140 or evals_log['train']['rmse'][-1]<1e-4
        for depth in [6,12]:
            cfg=dict(objective='reg:squarederror',max_depth=depth,eta=.05,lambda_=1)
            params={'objective':'reg:squarederror','max_depth':depth,'eta':.05,'lambda':1.,'subsample':1.,'colsample_bytree':1.,'tree_method':'hist','device':'cpu','seed':20260929,'nthread':2,'eval_metric':'rmse'}
            dm=xgb.DMatrix(xt,label=yt);de=xgb.DMatrix(xe)
            cb=Limit();t=time.monotonic()
            model=xgb.train(params,dm,num_boost_round=4000,evals=[(dm,'train')],callbacks=[cb],verbose_eval=False)
            elapsed=time.monotonic()-t
            pred_t=model.predict(dm)*ys+ym;pred_e=model.predict(de)*ys+ym
            name='xgboost-depth'+str(depth)
            model.save_model(out/(name+'.ubj'))
            np.savez_compressed(out/(name+'-predictions.npz'),train_row_ids=train,train_predictions=pred_t,test_row_ids=test,test_predictions=pred_e)
            result['models'][name]=dict(params=params,trees=model.num_boosted_rounds(),fit_seconds=elapsed,metrics=scores(pred_t,pred_e))
            atomic_json(out/'status.json',result)
            print(json.dumps({'model':name,**result['models'][name]}),flush=True)
        xx=torch.tensor(xt);yy=torch.tensor(yt).reshape(-1,1);tt=torch.tensor(xe)
        for width in [128,256]:
            torch.manual_seed(20260929)
            model=torch.nn.Sequential(torch.nn.Linear(32,width),torch.nn.ReLU(),torch.nn.Linear(width,width),torch.nn.ReLU(),torch.nn.Linear(width,width),torch.nn.ReLU(),torch.nn.Linear(width,1))
            optim=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0.)
            t=time.monotonic();epoch=0
            name='mlp-'+str(width)+'x3'
            while epoch<10000 and time.monotonic()-t<300:
                model.train();perm=torch.randperm(len(xx))
                for ids in perm.split(256):
                    optim.zero_grad(set_to_none=True)
                    loss=(model(xx[ids])-yy[ids]).square().mean()
                    assert torch.isfinite(loss)
                    loss.backward();optim.step()
                epoch+=1
                if epoch%10==0:
                    model.eval()
                    with torch.no_grad(): mse=float((model(xx)-yy).square().mean())
                    append_event(out/'mlp-trace.jsonl',dict(model=name,epoch=epoch,seconds=time.monotonic()-t,train_mse_standardized=mse))
                    if mse<1e-8:break
            elapsed=time.monotonic()-t
            model.eval()
            with torch.no_grad():
                pred_t=model(xx).squeeze(1).numpy().astype(np.float64)*ys+ym
                pred_e=model(tt).squeeze(1).numpy().astype(np.float64)*ys+ym
            assert np.isfinite(pred_t).all() and np.isfinite(pred_e).all()
            torch.save({'model':model.state_dict(),'width':width,'input_mean':xm,'input_std':xs,'target_mean':ym,'target_std':ys},out/(name+'.pt'))
            np.savez_compressed(out/(name+'-predictions.npz'),train_row_ids=train,train_predictions=pred_t,test_row_ids=test,test_predictions=pred_e)
            result['models'][name]=dict(architecture=[32,width,width,width,1],activation='ReLU',optimizer='AdamW',lr=.001,weight_decay=0.,batch_size=256,parameter_count=sum(p.numel() for p in model.parameters()),epochs=epoch,fit_seconds=elapsed,metrics=scores(pred_t,pred_e))
            atomic_json(out/'status.json',result)
            print(json.dumps({'model':name,**result['models'][name]}),flush=True)
        result.update(outcome='completed',seconds=time.monotonic()-started)
    except Exception as e:
        result.update(outcome='failed',error_type=type(e).__name__,error=str(e));atomic_json(out/'terminal.json',result);raise
    atomic_json(out/'status.json',result);atomic_json(out/'terminal.json',result)

if __name__=='__main__':run()
