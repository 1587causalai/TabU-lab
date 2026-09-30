"""Bounded replay of a failed backward, without changing data or optimizer state."""
import time
import torch

def backward_checked(forward, model, optimizer, *, capture_rng, restore_rng, sync, on_failure):
    saved_rng = capture_rng()
    failed_seconds = 0.0
    for attempt in range(2):
        attempt_started = time.monotonic()
        if attempt:
            restore_rng(saved_rng)
        optimizer.zero_grad(set_to_none=True)
        result = forward()
        loss_finite = bool(torch.isfinite(result.loss).all())
        if loss_finite:
            result.loss.backward()
        sync()
        grads = [p for p in model.parameters() if p.grad is not None]
        bad = {name: {'nan': int(torch.isnan(p.grad).sum()), 'inf': int(torch.isinf(p.grad).sum())}
               for name, p in model.named_parameters()
               if p.grad is not None and not bool(torch.isfinite(p.grad).all())}
        if loss_finite and grads and not bad:
            return result, grads, attempt, failed_seconds
        report = dict(attempt=attempt, loss_finite=loss_finite, loss=float(result.loss.detach()),
                      gradient_parameters=len(grads), bad_gradients=bad)
        del result, grads
        optimizer.zero_grad(set_to_none=True)
        restore_rng(saved_rng)
        on_failure(report)
        if attempt:
            raise FloatingPointError('nonfinite loss/gradients persisted after one identical replay')
        sync()
        device = next(model.parameters()).device
        if device.type == 'mps':
            torch.mps.empty_cache()
        failed_seconds += time.monotonic() - attempt_started
    raise AssertionError('unreachable')
