"""Build the restoration dashboard; persist only with explicit --save."""
import argparse
import json

import wandb_workspaces.reports.v2 as wr
from wandb_workspaces import expr


def build_report(entity, run_id):
    metrics = [
        ("Training: loss over 120 tables", "train_round/loss"),
        ("Query numeric: encoding MSE", "evaluation/query/numeric/encoding_mse"),
        ("Query discrete: encoding MSE", "evaluation/query/discrete/encoding_mse"),
        ("Query discrete: error rate", "evaluation/query/discrete/discrete_error_rate"),
        ("Retained numeric: encoding MSE", "evaluation/retained/numeric/encoding_mse"),
        ("Retained discrete: encoding MSE", "evaluation/retained/discrete/encoding_mse"),
        ("Retained discrete: error rate", "evaluation/retained/discrete/discrete_error_rate"),
    ]
    runset = wr.Runset(
        entity=entity, project="restoration", name="Global 2.5% with numeric tail protection",
        filters=[expr.Metric("ID") == run_id],
    )
    panels = [wr.LinePlot(
        title=title, x="completed_round",
        y=[prefix + "/" + statistic for statistic in ("mean", "median", "p95")],
        title_x="Completed rounds", smoothing_type="none", aggregate=False,
        layout=wr.Layout(x=(index % 2) * 12, y=(index // 2) * 8, w=12, h=8),
    ) for index, (title, prefix) in enumerate(metrics)]
    for index, (title, metric) in enumerate((
        ("Gradient norm before clipping", "train/gradient_norm"),
        ("Gradient norm after clipping (clip=1)", "train/post_clip_gradient_norm"),
        ("Protected numeric tails selected as Query (must be zero)",
         "train/coverage/query_numeric_tail_cells"),
    ), start=len(panels)):
        panels.append(wr.LinePlot(
            title=title, x="update", y=[metric], title_x="Optimizer updates",
            smoothing_type="none", aggregate=False,
            layout=wr.Layout(x=(index % 2) * 12, y=(index // 2) * 8, w=12, h=8),
        ))
    return wr.Report(
        entity=entity, project="restoration", width="fluid",
        title="Restoration | tail guard and gradient monitoring",
        description=(
            "Training statistics cover one complete 120-table round. "
            "Evaluation statistics give each table equal weight after pooling "
            "8 fixed masks per table. Query and retained are shown separately. "
            "Numeric tails above 8 half-IQR from the training-pool median remain visible. "
            "Query metrics are conditional on eligibility. Reserved rows are excluded."
        ),
        blocks=[wr.PanelGrid(runsets=[runset], panels=panels)],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="zj3712")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    report = build_report(args.entity, args.run_id)
    if args.save:
        report.save()
        print(report.url)
    else:
        # Force full schema validation without submitting anything to W&B.
        model = report._to_model()
        print(json.dumps({
            "saved": False, "panels": len(report.blocks[0].panels),
            "serialized_bytes": len(model.model_dump_json()),
            "run_filter": report.blocks[0].runsets[0]._to_model().filters,
        }, sort_keys=True))
