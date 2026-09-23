"""Add historical and incremental balanced-cycle loss to an existing outbox."""
from cycle_loss import (initial_state, advance, initial_replay_state, advance_replay,
                        REPLAY_GROUPS, WEIGHTED_REPLAY_KIND)


def replay_enabled(config):
    policy = config.get("loss_replay")
    if policy is None:
        return False
    if not isinstance(policy, dict) or policy.get("kind") not in REPLAY_GROUPS:
        raise ValueError("unsupported loss replay policy")
    return True


class CycleMixin:
    def cycle_request(self, config):
        expected_schema = ("normal120-replay-median-p05-p95-linear-v2" if replay_enabled(config)
                           else "median-p05-p95-linear-v1")
        if config.get("loss_replay", {}).get("kind") == WEIGHTED_REPLAY_KIND:
            expected_schema = "normal120-replay42-median-p05-p95-p99-linear-v3"
        schema = self.get("cycle_distribution_schema")
        if schema is None:
            # Preserve previously delivered means. Re-read only the raw-cycle
            # cursor to append the newly requested distribution metrics.
            with self.db:
                self.put("cycle_mean_before_distribution_backfill", {
                    key: self.get(key) for key in ("cycle_state", "cycle_offset", "cycle_last_update", "cycle_complete_count")})
                self.put("cycle_state", None)
                self.put("cycle_offset", 0)
                self.put("cycle_last_update", config.get("initial_update", 0))
                self.put("cycle_complete_count", 0)
                self.put("cycle_distribution_schema", expected_schema)
        elif schema != expected_schema:
            raise ValueError("unsupported cycle distribution schema")
        return {"output": config["remote_output"],
                "offset": self.get("cycle_offset", 0),
                "last_update": self.get("cycle_last_update", config.get("initial_update", 0)),
                "seen_evaluations": [], "cycles_only": True}

    def ingest_cycles(self, snapshot, config, references):
        if snapshot["identity_sha256"] != config["identity_sha256"] or snapshot["source_sha256"] != config["source_sha256"]:
            raise ValueError("cycle stream identity mismatch")
        if set(snapshot["tables"]) != set(references) or len(snapshot["tables"]) != 120:
            raise ValueError("cycle stream table scope mismatch")
        base = config.get("initial_update", 0)
        replay = replay_enabled(config)
        state = self.get("cycle_state")
        if state is None:
            state = (initial_replay_state(sorted(references), base, config["initial_normal_update"],
                                          policy_kind=config["loss_replay"]["kind"])
                     if replay else initial_state(sorted(references), base_update=base))
        # Pure advance rejects the entire batch before any outbox state changes.
        new_state, cycles = (advance_replay if replay else advance)(state, snapshot["updates"])
        with self.db:
            for cycle in cycles:
                update = cycle["end_new_update"]
                axes = ({"train_cycle/normal_update": cycle["end_normal_update"],
                         "train_cycle/new_normal_update": update,
                         "train_cycle/new_actual_update": cycle["end_global_update"] - base,
                         "train_cycle/new_extra_updates": cycle["end_global_update"] - base - update,
                         "train_cycle/cumulative_extra_updates": cycle["end_global_update"] - cycle["end_normal_update"]}
                        if replay else {})
                self.emit("cycle_loss:" + str(cycle["end_global_update"]), {
                    **axes,
                    "train_cycle/update": update,
                    "train_cycle/global_update": cycle["end_global_update"],
                    "train_cycle/index": cycle["cycle_index"],
                    "train_cycle/mean_loss": cycle["mean_loss"],
                    "train_cycle/table_count": cycle["table_count"],
                    "train_cycle/step_count": 120,
                })
                self.emit("cycle_distribution:" + str(cycle["end_global_update"]), {
                    **axes,
                    "train_cycle/update": update,
                    "train_cycle/global_update": cycle["end_global_update"],
                    "train_cycle/index": cycle["cycle_index"],
                    "train_cycle/median_loss": cycle["median_loss"],
                    "train_cycle/p05_loss": cycle["p05_loss"],
                    "train_cycle/p95_loss": cycle["p95_loss"],
                    "train_cycle/table_count": cycle["table_count"],
                    **({"train_cycle/p99_loss": cycle["p99_loss"]} if "p99_loss" in cycle else {}),
                })
            self.put("cycle_state", new_state)
            self.put("cycle_offset", snapshot["offset"])
            last = snapshot["updates"][-1]["update"] if snapshot["updates"] else self.get("cycle_last_update", base)
            self.put("cycle_last_update", last)
            self.put("cycle_complete_count", new_state["new_updates"] // 120)

    def cycle_status(self, config):
        base = config.get("initial_update", 0)
        last = self.get("cycle_last_update", base)
        state = self.get("cycle_state")
        normal_processed = state["new_updates"] if state is not None else 0
        replay = replay_enabled(config)
        result = {"cycle_last_global_update": last,
                "cycle_distribution_schema": self.get("cycle_distribution_schema"),
                "cycle_complete_count": self.get("cycle_complete_count", 0),
                "cycle_partial_steps": (normal_processed if replay else last - base) % 120}
        if replay:
            normal_absolute = state["normal_update"] if state else config["initial_normal_update"]
            result.update(cycle_last_normal_update=(state["normal_update"] if state else config["initial_normal_update"]),
                          cycle_new_normal_updates=normal_processed,
                          cycle_extra_updates=(state["extra_updates"] if state else {
                              kind: 0 for kind, _, _ in REPLAY_GROUPS[config["loss_replay"]["kind"]]}),
                          cycle_new_extra_updates=last - base - normal_processed,
                          cycle_cumulative_extra_updates=last - normal_absolute,
                          cycle_next_phase=state["phase"] if state else "normal")
            if state and "phase_pass" in state:
                result["cycle_extra_pass_index"] = state["phase_pass"]
        return result

    def cycles_caught_up_to_terminal(self, config):
        terminal = self.get("terminal")
        return terminal is not None and self.get("cycle_last_update", config.get("initial_update", 0)) >= terminal["update"]
