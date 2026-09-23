"""Pure classification of a training receipt versus telemetry close status.

Finished telemetry does not imply a completed training budget or a fit verdict.
The SSH reader supplies checkpoint byte-integrity evidence; this module never
loads tensors, contacts W&B, or changes a process.
"""
import re

TERMINAL_SUMMARY_FIELDS = (
    "outcome", "update", "durable_update", "checkpoint_sha256", "total_seconds",
    "error_type", "error_present", "terminal_identity_matches", "checkpoint_verified",
    "checkpoint_verification", "normal_update", "extra_updates",
)


def classify_terminal(terminal, config):
    """Return the W&B exit code and explicit, scalar-only summary fields.

    Only the runner's saved boundary stops qualify as ``closed_after_stop``.
    Exception-path KeyboardInterrupt also says interrupted, but its error_type
    prevents normal closure. Unknown/budget_exhausted/inconclusive remain failed.
    """
    outcome = terminal.get("outcome")
    update, durable = terminal.get("update"), terminal.get("durable_update")
    valid_update = type(update) is int and update >= 0
    has_error = (terminal.get("error_present") is True
                 or terminal.get("error") not in (None, "")
                 or terminal.get("error_type") not in (None, ""))
    checkpoint_verified = (terminal.get("checkpoint_verified") is True
                           and isinstance(terminal.get("checkpoint_sha256"), str)
                           and re.fullmatch(r"[0-9a-f]{64}", terminal["checkpoint_sha256"]) is not None)
    identity_matches = terminal.get("terminal_identity_matches")
    non_durable = durable is not None and (type(durable) is not int or durable != update)
    safe_stop = (outcome in ("stopped", "interrupted") and not has_error
                 and valid_update and type(durable) is int and durable == update
                 and checkpoint_verified and identity_matches is True)
    completed = (outcome == "completed" and not has_error and valid_update
                 and not non_durable and identity_matches is not False)
    if has_error:
        reason = "training_error_recorded"
    elif not valid_update:
        reason = "invalid_terminal_update"
    elif identity_matches is False:
        reason = "terminal_identity_mismatch"
    elif non_durable:
        reason = "terminal_update_not_durable"
    elif safe_stop:
        reason = "durable_checkpoint_stop_verified"
    elif completed:
        reason = "completed_terminal_without_error"
    elif outcome in ("stopped", "interrupted"):
        reason = "stop_durability_not_verified"
    else:
        reason = "unrecognized_or_unsuccessful_training_outcome"
    exit_code = 0 if completed or safe_stop else 1
    expected = config.get("global_max_updates")
    if expected is None and all(type(config.get(k)) is int for k in ("initial_update", "new_max_updates")):
        expected = config["initial_update"] + config["new_max_updates"]
    budget_completed = completed and type(expected) is int and update == expected
    policy = config.get("loss_replay")
    if policy:
        budget_completed = (budget_completed and type(terminal.get("normal_update")) is int
                            and terminal["normal_update"] == policy.get("normal_max_updates"))
    lifecycle = "closed_after_completion" if completed else "closed_after_stop" if safe_stop else "failed_training_terminal"
    return {"exit_code": exit_code, "summary": {
        "training_terminal_confirmed": True,
        "training_outcome": outcome if isinstance(outcome, str) else "unknown",
        "training_intentionally_stopped": safe_stop,
        "training_budget_completed": bool(budget_completed),
        "training_checkpoint_verified": checkpoint_verified,
        "telemetry/lifecycle": lifecycle,
        "telemetry/finish_exit_code": exit_code,
        "telemetry/termination_reason": reason,
    }}


def finish_telemetry(run, terminal, config):
    """Close only the supplied observer run; preserve the training outcome."""
    decision = classify_terminal(terminal, config)
    run.summary.update({"terminal/" + k: terminal[k] for k in TERMINAL_SUMMARY_FIELDS
                        if terminal.get(k) is not None})
    run.summary.update(decision["summary"])
    run.finish(exit_code=decision["exit_code"])
    return decision
