"""Validation-only stopping selection, direct selected-model test; no refit."""
import copy
import hashlib
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb

B = Path(sys.argv[sys.argv.index('--root')+1]).resolve()
OLD = B.parent.parent.parent / 'v6-puma-single-hostloss-15m-20260929'
sys.path.insert(0, str(OLD / 'source' / 'src'))
sys.path.insert(0, str(OLD))
from common import metric_rows
def load_data():
    return json.loads((B/'fit-data.json').read_text()), json.loads((B/'fit-bank.json').read_text())
WIDTH=len(load_data()[0]['values'][0])-1
OUT=B/'baselines'
from tabu_lab.curriculum_v53.artifacts import atomic_json, append_event, sha256

SEED = 20260929


def utc():
    return datetime.now(timezone.utc).isoformat()


def normalize(x, y, ids):
    xm, xs = x[ids].mean(0), x[ids].std(0)
    xs[xs < 1e-12] = 1
    ym, ys = y[ids].mean(), y[ids].std()
    assert ys > 0
    return dict(xm=xm, xs=xs, ym=ym, ys=ys)


def arrays(x, y, ids, norm):
    return (((x[ids] - norm['xm']) / norm['xs']).astype(np.float32),
            ((y[ids] - norm['ym']) / norm['ys']).astype(np.float32))


def mlp(width):
    torch.manual_seed(SEED)
    return torch.nn.Sequential(torch.nn.Linear(WIDTH, width), torch.nn.ReLU(),
        torch.nn.Linear(width, width), torch.nn.ReLU(),
        torch.nn.Linear(width, width), torch.nn.ReLU(), torch.nn.Linear(width, 1))


def epoch(model, optimizer, xx, yy):
    model.train()
    for ids in torch.randperm(len(xx)).split(256):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(xx[ids]) - yy[ids]).square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()


class TimeLimit(xgb.callback.TrainingCallback):
    def before_training(self, model):
        self.started = time.monotonic()
        return model

    def after_iteration(self, model, epoch, evals_log):
        return time.monotonic() - self.started >= 140


def run():
    OUT.mkdir(exist_ok=False)
    (OUT/'artifacts').mkdir()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    data, bank = load_data()
    raw = np.asarray(data['values'], dtype=np.float64)
    x, y = raw[:, :-1], raw[:, -1]
    split=json.loads((B/'split.json').read_text())
    fit=np.asarray(split['fit_row_ids']);valid=np.asarray(split['validation_row_ids']);test=np.asarray(split['test_row_ids'])
    train=fit
    assert set(fit).isdisjoint(valid) and set(fit).isdisjoint(test) and set(valid).isdisjoint(test)
    norm = normalize(x, y, fit)
    xf, yf = arrays(x, y, fit, norm)
    xv, yv = arrays(x, y, valid, norm)
    selected, models = {}, {}
    fitted = {}
    status = dict(outcome='selecting', started_utc=utc(), device='cpu', dtype='float32', threads=2,
        seed=SEED, bank_sha256=sha256(B / 'fit-bank.json'), split_sha256=sha256(B / 'split.json'),
        train_n=len(train), test_n=len(test), fit_n=len(fit), validation_n=len(valid),
        script_sha256=sha256(Path(__file__)), old_baseline_script_sha256=sha256(OLD / 'baselines.py'),
        source_common_sha256=sha256(OLD / 'common.py'), versions=dict(python=platform.python_version(),
        numpy=np.__version__, torch=torch.__version__, xgboost=xgb.__version__), selected=selected, models=models)

    def save():
        atomic_json(OUT / 'status.json', status)

    def record(name, **kw):
        append_event(OUT / 'trace.jsonl', dict(model=name, utc=utc(), **kw))

    def score(ids, predictions):
        return metric_rows([dict(row_id=int(i), target=float(y[i]), prediction=float(p))
                            for i, p in zip(ids, predictions, strict=True)], data)

    save()
    try:
        # One fixed configuration per family; no test predictions in selection.
        for depth in [6]:
            name = f'xgboost-depth{depth}'
            status['active_model'] = name
            save()
            params = dict(objective='reg:squarederror', max_depth=depth, eta=.05,
                reg_lambda=1., subsample=1., colsample_bytree=1., tree_method='hist',
                device='cpu', seed=SEED, nthread=2, eval_metric='rmse')
            history = {}
            tick = time.monotonic()
            model = xgb.train(params, xgb.DMatrix(xf, label=yf), num_boost_round=4000,
                evals=[(xgb.DMatrix(xv, label=yv), 'validation')],
                early_stopping_rounds=100, callbacks=[TimeLimit()], evals_result=history, verbose_eval=False)
            best = int(model.best_iteration) + 1
            count = model.num_boosted_rounds()
            vp = model.predict(xgb.DMatrix(xv), iteration_range=(0, best)) * norm['ys'] + norm['ym']
            selected[name] = dict(params=params, best_rounds=best, searched_rounds=count,
                patience=100, selection_seconds=time.monotonic()-tick,
                stopped_by_patience=count-best >= 100, validation_metrics=score(valid, vp))
            (OUT/'artifacts'/f'{name}-validation-history.json').write_text(json.dumps(history))
            fitted[name] = model[:best]
            model[:best].save_model(OUT/'artifacts'/f'{name}-selected.ubj')
            record(name, phase='selection', **selected[name])
            save()
            print(json.dumps(dict(event='selected', model=name, **selected[name])), flush=True)

        xx, yy = torch.tensor(xf), torch.tensor(yf).reshape(-1, 1)
        vx, vy = torch.tensor(xv), torch.tensor(yv).reshape(-1, 1)
        for width in [256]:
            name = f'mlp-{width}x3'
            status['active_model'] = name
            save()
            model = mlp(width)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0.)
            tick = time.monotonic()
            best, best_epoch, count, best_state = float('inf'), 0, 0, None
            while count < 10000 and time.monotonic()-tick < 300:
                epoch(model, optimizer, xx, yy)
                count += 1
                model.eval()
                with torch.no_grad():
                    loss = float((model(vx)-vy).square().mean())
                assert math.isfinite(loss)
                if loss < best:
                    best, best_epoch, best_state = loss, count, copy.deepcopy(model.state_dict())
                record(name, phase='selection', epoch=count, validation_mse_standardized=loss,
                       best_epoch=best_epoch, seconds=time.monotonic()-tick)
                if count-best_epoch >= 50:
                    break
            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                vp = model(vx).squeeze(1).numpy().astype(np.float64)*norm['ys']+norm['ym']
            selected[name] = dict(width=width, best_rounds=best_epoch, searched_rounds=count,
                patience=50, selection_seconds=time.monotonic()-tick,
                stopped_by_patience=count-best_epoch >= 50, validation_metrics=score(valid, vp))
            fitted[name] = model
            torch.save(dict(model=best_state, normalization=norm, width=width), OUT/'artifacts'/f'{name}-selected.pt')
            save()
            print(json.dumps(dict(event='selected', model=name, **selected[name])), flush=True)

        status['selection_completed_utc'] = utc()
        atomic_json(OUT/'selection.json', dict(completed_utc=status['selection_completed_utc'], selected=selected))
        full_norm=norm
        xt,yt=xf,yf
        for name,choice in selected.items():
            models[name]=dict(selected_rounds=choice['best_rounds'],refit=False)
        # Test scoring happens only after both stopping choices are frozen.
        status.update(outcome='evaluating', test_started_utc=utc())
        save()
        xe, _ = arrays(x, y, test, full_norm)
        for name, model in fitted.items():
            if name.startswith('xgboost'):
                pt, pe = model.predict(xgb.DMatrix(xt)), model.predict(xgb.DMatrix(xe))
            else:
                model.eval()
                with torch.no_grad():
                    pt = model(torch.tensor(xt)).squeeze(1).numpy()
                    pe = model(torch.tensor(xe)).squeeze(1).numpy()
            pt, pe = (a.astype(np.float64)*full_norm['ys']+full_norm['ym'] for a in (pt, pe))
            assert np.isfinite(pt).all() and np.isfinite(pe).all()
            models[name]['metrics'] = dict(train=score(train, pt), test=score(test, pe))
            np.savez_compressed(OUT/'artifacts'/f'{name}-predictions.npz', train_row_ids=train,
                train_predictions=pt, test_row_ids=test, test_predictions=pe)
            save()
            print(json.dumps(dict(event='scored', model=name, **models[name])), flush=True)
        status.update(outcome='completed', completed_utc=utc(), test_predictions_computed=True,
                      normalization_fit_only={k:np.asarray(v).tolist() for k,v in full_norm.items()})
    except Exception as e:
        status.update(outcome='failed', error_type=type(e).__name__, error=str(e))
        save()
        atomic_json(OUT/'terminal.json', status)
        raise
    save()
    atomic_json(OUT/'terminal.json', status)


if __name__ == '__main__':
    run()
