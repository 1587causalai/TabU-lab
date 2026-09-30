"""DGX2 OpenML12 shared-checkpoint trial: broadcast with V5.5 target loss."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import torch

from tabu_lab.curriculum_v53.artifacts import (
    atomic_json, append_event, finite_state, load_checkpoint, restore_rng as base_restore_rng, rng_state as base_rng_state, sha256,
)
from tabu_lab.curriculum_v53.data import build_episode
from tabu_lab.curriculum_v53.protocol import load_v55_plan, schedule_entry
from tabu_lab.curriculum_v53.runner import configure_runtime
from tabu_lab.models.restoration._dtype import execution_dtype
from tabu_lab.models.restoration_v53 import V53LossConfig
from tabu_lab.models.restoration_v6 import V6Model, score_training_episode
from tabu_lab.restoration_optimizers import adamw


def rng_state():
    result = base_rng_state()
    if torch.backends.mps.is_available():
        result["mps"] = torch.mps.get_rng_state()
    return result


def restore_rng(state):
    base_restore_rng(state)
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"])


def source_identity():
    own = Path(__file__).resolve().parent
    files = [__file__, *sorted((own / "source/src/tabu_lab/models/restoration_v6").glob("*.py"))]
    hashes = {str(Path(path).relative_to(own)): sha256(path) for path in files}
    return hashes


def new_identity(args, plan, runtime):
    record = dict(
        schema="tabu.v6.target-broadcast.target-only.v1",
        parent_sha256=args.parent_sha,
        parent_identity_sha256=plan.identity["sha256"],
        parent_manifest_sha256=sha256(args.manifest),
        model_config=plan.config.as_dict(),
        code=source_identity(),
        supervision="target_only",
        objective="V5.5 Query target-column squared loss with target broadcasting",
        sampling=plan.spec["stages"][0]["sampling"],
        recipe=plan.spec["stages"][0]["recipe"],
        device=args.device, runtime=runtime,
    )
    record["sha256"] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    return record


def checkpoint(root, model, optimizer, state, identity):
    if not finite_state(model.state_dict()) or not finite_state(optimizer.state_dict()):
        raise FloatingPointError("nonfinite state blocks V6 checkpoint")
    payload = dict(schema="tabu.v6.weights-only-checkpoint.v1", purpose="training",
                   identity=identity, model_config=model.config.as_dict(),
                   model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                   optimizer=optimizer.state_dict(), state=state, rng=rng_state())
    folder = root / "checkpoints"
    folder.mkdir(exist_ok=True)
    temporary = folder / ".pending.pt"
    torch.save(payload, temporary)
    digest = sha256(temporary)
    stable = folder / (digest + ".pt")
    os.replace(temporary, stable)
    alias = root / "checkpoint-progress.pt"
    link = root / ".checkpoint-progress.pt.tmp"
    link.symlink_to(Path("checkpoints") / stable.name)
    os.replace(link, alias)
    return stable, digest


def make_model(args, plan, payload):
    if payload["identity"] != plan.identity or payload["model_config"] != plan.config.as_dict():
        raise ValueError("parent checkpoint identity/config disagrees with frozen manifest")
    model = V6Model(plan.config, supervision="target_only").to(
        device=args.device, dtype=execution_dtype(args.device)
    )
    expected = model.state_dict()
    if set(payload["model"]) != set(expected):
        raise ValueError("V5.5 to V6 model tensor names differ")
    for name, value in payload["model"].items():
        if value.shape != expected[name].shape:
            raise ValueError(f"V5.5 to V6 tensor shape differs: {name}")
    model.load_state_dict(payload["model"], strict=True)
    for name, actual in model.state_dict().items():
        if not torch.equal(actual.cpu(), payload["model"][name].to(actual.dtype)):
            raise ValueError(f"weights-only copy differs: {name}")
    return model


def episode_seeds(plan):
    seeds = dict(plan.spec["seeds"])
    # Preserve the previous V6 episode stream for a matched objective comparison.
    namespace = "v6-target-broadcast-dgx2-20260927"
    for stream in ("masks", "codes", "windows"):
        seeds[stream] = int.from_bytes(
            hashlib.sha256(f"{seeds[stream]}/{namespace}".encode()).digest()[:8], "little"
        )
    return seeds


def main(args):
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=False)
    receipt = dict(schema="tabu.v6.broadcast-target-only-campaign.v1", outcome="starting",
                   host=args.host, parent=str(Path(args.parent).resolve()),
                   parent_sha256=args.parent_sha, manifest=str(Path(args.manifest).resolve()),
                   continuation=("strict_resume_optimizer_rng_cursor_exposure" if args.resume_checkpoint
                                 else "weights_only_broadcast_old_loss_new_optimizer_rng_cursor_exposure"),
                   successful_update_seconds_target=args.seconds,
                   focus_tables=["pumadyn32nh", "kin8nm"])
    atomic_json(root / "campaign.json", receipt)
    wandb_run = None
    try:
        runtime = configure_runtime(args.device)
        plan = load_v55_plan(args.manifest)
        stage = plan.spec["stages"][0]
        if any(item["kind"] != "supervised_row" for item in stage["recipe"].values()) or len(plan.tables) != 630:
            raise ValueError("V6 trial requires frozen 630-table supervised-row plan")
        loss_config = V53LossConfig(**stage["loss"])
        if tuple(loss_config.state_weights) != (0.0, 1.0, 0.0, 0.0):
            raise ValueError("target-only trial requires the original Query-only squared loss")
        payload, digest = load_checkpoint(args.parent)
        if digest != args.parent_sha or payload["purpose"] != "training":
            raise ValueError("parent checkpoint SHA/purpose differs")
        model = make_model(args, plan, payload)
        parent_update = payload["state"]["update"]
        del payload
        identity = new_identity(args, plan, runtime)
        random.seed(plan.spec["seeds"]["model"] + 600)
        torch.manual_seed(plan.spec["seeds"]["model"] + 600)
        torch.cuda.manual_seed_all(plan.spec["seeds"]["model"] + 600)
        optimizer = adamw(model, plan.optimizer)
        state = dict(update=0, cursor=0, successful_update_seconds=0.0,
                     exposure={"focus2":0,"other10":0,"history618":0},
                     parent_update=parent_update,
                     rng_seed=plan.spec["seeds"]["model"] + 600)
        if args.resume_checkpoint:
            if sha256(args.resume_checkpoint) != args.resume_sha:
                raise ValueError("V6 resume checkpoint SHA mismatch")
            previous = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
            if (previous["schema"] != "tabu.v6.weights-only-checkpoint.v1"
                    or previous["identity"] != identity
                    or previous["model_config"] != plan.config.as_dict()
                    or previous["state"]["parent_update"] != parent_update):
                raise ValueError("V6 strict resume identity/config/parent drift")
            model.load_state_dict(previous["model"], strict=True)
            optimizer.load_state_dict(previous["optimizer"])
            state = previous["state"]
            restore_rng(previous["rng"])
            receipt.update(resume_checkpoint=str(Path(args.resume_checkpoint).resolve()),
                           resume_sha256=args.resume_sha,
                           resume_start_update=state["update"],
                           resume_start_successful_seconds=state["successful_update_seconds"])
        initial_path, initial_sha = checkpoint(root, model, optimizer, state, identity)
        receipt.update(outcome="admitted", runtime=runtime, identity=identity,
                       parent_update=parent_update, initial_checkpoint=str(initial_path),
                       initial_checkpoint_sha256=initial_sha)
        atomic_json(root / "campaign.json", receipt)

        try:
            import wandb
            wandb_settings = wandb.Settings(init_timeout=30, disable_git=True,
                                            disable_code=True, console="off")
            wandb_run = wandb.init(entity="zj3712", project="restoration-v6-target-broadcast",
                                   id="v6broadcastoldloss-dgx2-20260927", resume="allow",
                                   name="dgx2-H8-broadcast-oldloss-openml12", mode="online",
                                   settings=wandb_settings, config=dict(
                                       identity=identity["sha256"], parent_sha256=digest,
                                       device=args.device, seconds=args.seconds,
                                       sampling="focus2:6,other10:10,history618:4"))
            receipt["wandb_status"] = "online"
            receipt["wandb_url"] = wandb_run.url
            atomic_json(root / "campaign.json", receipt)
        except Exception as error:
            receipt.update(wandb_status="unavailable", wandb_error_type=type(error).__name__)
            atomic_json(root / "campaign.json", receipt)
            wandb_run = None

        seeds = episode_seeds(plan)
        last_save = state["successful_update_seconds"]
        while state["successful_update_seconds"] < args.seconds:
            table, episode_index = schedule_entry(plan, 0, state["cursor"])
            t0 = time.monotonic()
            inputs, request, truth, info = build_episode(
                table, stage["recipe"][table.kind], episode_index, seeds, args.device,
                epsilon=plan.config.epsilon, codec_version=plan.config.codec_version,
            )
            model.train()
            optimizer.zero_grad(set_to_none=True)
            score = score_training_episode(
                model, inputs, truth, request=request, loss_config=loss_config
            )
            score.loss.backward()
            gradients = [p for p in model.parameters() if p.grad is not None]
            if not gradients or not finite_state([p.grad for p in gradients]):
                raise FloatingPointError("missing or nonfinite V6 gradients")
            norm = torch.nn.utils.clip_grad_norm_(
                gradients, plan.optimizer.grad_clip, error_if_nonfinite=True
            )
            optimizer.step()
            if not finite_state(model.state_dict()) or not finite_state(optimizer.state_dict()):
                raise FloatingPointError("nonfinite V6 state after update")
            torch.mps.synchronize() if args.device == "mps" else torch.cuda.synchronize()
            seconds = time.monotonic() - t0
            state["successful_update_seconds"] += seconds
            state["update"] += 1
            state["cursor"] += 1
            state["exposure"][table.cohort] += 1
            row = dict(update=state["update"], cohort=table.cohort, table=table.name,
                       episode_index=episode_index, loss=float(score.loss.detach()),
                       gradient_norm=float(norm), seconds=seconds,
                       successful_update_seconds=state["successful_update_seconds"],
                       query_rows=score.query_rows, scored_cells=score.scored_cells,
                       hidden_target_cells=info["query_count"])
            append_event(root / "updates.jsonl", row)
            if wandb_run:
                wandb_run.log({"train/loss":row["loss"],
                               "train/gradient_norm":row["gradient_norm"],
                               "train/successful_seconds":row["successful_update_seconds"],
                               f"exposure/{table.cohort}":state["exposure"][table.cohort]},
                              step=state["update"])
            if state["successful_update_seconds"] - last_save >= args.save_every_seconds:
                path, checksum = checkpoint(root, model, optimizer, state, identity)
                receipt.update(outcome="running", checkpoint=str(path),
                               checkpoint_sha256=checksum, checkpoint_update=state["update"],
                               successful_update_seconds=state["successful_update_seconds"],
                               cohort_updates=state["exposure"])
                atomic_json(root / "campaign.json", receipt)
                last_save = state["successful_update_seconds"]
                print(json.dumps({"update":state["update"], "seconds":round(last_save,2),
                                  "loss":row["loss"], "sha256":checksum}), flush=True)
        path, checksum = checkpoint(root, model, optimizer, state, identity)
        receipt.update(outcome="training_completed", checkpoint=str(path),
                       checkpoint_sha256=checksum, checkpoint_update=state["update"],
                       successful_update_seconds=state["successful_update_seconds"],
                       cohort_updates=state["exposure"])
        atomic_json(root / "campaign.json", receipt)
        print(json.dumps({"outcome":receipt["outcome"], "update":state["update"],
                          "seconds":state["successful_update_seconds"],
                          "checkpoint_sha256":checksum}), flush=True)
    except Exception as error:
        receipt.update(outcome="failed", error_type=type(error).__name__, error=str(error))
        atomic_json(root / "campaign.json", receipt)
        raise
    finally:
        if wandb_run:
            wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("root", "manifest", "parent", "parent-sha", "device"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--seconds", type=float, default=3600.0)
    parser.add_argument("--save-every-seconds", type=float, default=300.0)
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--resume-sha")
    args = parser.parse_args()
    if bool(args.resume_checkpoint) != bool(args.resume_sha):
        parser.error("resume checkpoint and SHA must be provided together")
    main(args)
