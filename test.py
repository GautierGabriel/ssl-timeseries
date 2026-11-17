import os
import argparse
import logging
import json
import torch
import numpy as np
import pandas as pd
from typing import Optional

from datetime import datetime

from vivml.SSL.config_yml import parse_config
from vivml.SSL.model.builder import build_model_from_config
from vivml.SSL.data_streamer.wrapper import stream_wrapper
from vivml.SSL.data_streamer.builder import get_dataloaders
from vivml.SSL.data_streamer.source import DataSource
from vivml.SSL.evaluation.evaluation import AnomalyEvaluation, RepresentationEvaluation, SupervisedEvaluation
from vivml.SSL.evaluation.visualizer import DownstreamTaskVisualizer, AnomalyVisualizer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def reconstruct_plant_signal(group: pd.DataFrame, sampling_rate_sec: float) -> pd.DataFrame:
    freq = pd.Timedelta(seconds=sampling_rate_sec)
    series_list = []

    for _, row in group.sort_values("timestamp").iterrows():
        w = row["_raw_window"]
        start_ts = pd.to_datetime(row["timestamp"])
        idx = pd.date_range(start=start_ts, periods=len(w), freq=freq)
        series_list.append(pd.Series(np.asarray(w), index=idx))

    if not series_list:
        return pd.DataFrame(columns=["mV"])

    combined = pd.concat(series_list)
    combined = combined.groupby(combined.index).mean().sort_index()

    full_idx = pd.date_range(start=combined.index.min(), end=combined.index.max(), freq=freq)
    combined = combined.reindex(full_idx)
    return pd.DataFrame({"mV": combined.values}, index=combined.index)


def build_plant_reports(df_ml: pd.DataFrame, output_dir: str, top_n_plants: int,
                        context_days: float, sampling_rate_sec: float,
                        can_reconstruct_signal: bool) -> None:
    plant_reports_dir = os.path.join(output_dir, "plant_reports")
    os.makedirs(plant_reports_dir, exist_ok=True)
    csv_cols = [c for c in df_ml.columns if c != "_raw_window"]

    agg_dict = {"score": ["mean", "max", "min", "count"]}
    if "label" in df_ml.columns:
        agg_dict["label"] = ["mean"]

    plant_summary = df_ml.groupby("channel_id").agg(agg_dict)
    plant_summary.columns = ["_".join(filter(None, c)).strip("_") for c in plant_summary.columns]

    if "label_mean" in plant_summary.columns:
        plant_summary = plant_summary.rename(columns={"label_mean": "label_stress_ratio"})

    plant_summary = plant_summary.sort_values("score_max", ascending=False)
    plant_summary.to_csv(os.path.join(plant_reports_dir, "plant_summary.csv"))
    logger.info(f"Plant summary saved -> plant_reports/plant_summary.csv")

    for plant_id, group in df_ml.groupby("channel_id"):
        plant_dir = os.path.join(plant_reports_dir, str(plant_id))
        os.makedirs(plant_dir, exist_ok=True)
        group.sort_values("timestamp")[csv_cols].to_csv(os.path.join(plant_dir, "scores.csv"), index=False)

    top_plants = plant_summary.head(top_n_plants).index.tolist()
    logger.info(f"Top {len(top_plants)} anomalous plants: {top_plants}")

    if not can_reconstruct_signal:
        logger.warning("Raw signal reconstruction skipped (no raw data available).")
        return

    context = pd.Timedelta(days=context_days)

    for plant_id in top_plants:
        group = df_ml[df_ml["channel_id"] == plant_id].sort_values("timestamp")
        plant_dir = os.path.join(plant_reports_dir, str(plant_id))

        df_signal = reconstruct_plant_signal(group, sampling_rate_sec=sampling_rate_sec)

        peak_ts = pd.to_datetime(group.loc[group["score"].idxmax(), "timestamp"])
        crop_start, crop_end = peak_ts - context, peak_ts + context

        df_signal_cropped = df_signal.loc[crop_start:crop_end]
        ts = pd.to_datetime(group["timestamp"])
        group_cropped = group[(ts >= crop_start) & (ts <= crop_end)]

        AnomalyVisualizer.plot_signal_with_heatmap(
            df_signal=df_signal_cropped,
            df_ml=group_cropped[csv_cols],
            plant_id=str(plant_id),
            output_html_path=os.path.join(plant_dir, "signal_heatmap.html")
        )
        logger.info(f"Signal heatmap saved -> plant_reports/{plant_id}/signal_heatmap.html")


def build_trial_reports(df_ml: pd.DataFrame, output_dir: str, resample_freq: str = '6h') -> None:
    trial_reports_dir = os.path.join(output_dir, "trial_reports")
    os.makedirs(trial_reports_dir, exist_ok=True)

    if "label" not in df_ml.columns or "prob_stress" not in df_ml.columns:
        logger.warning("Labels or prob_stress missing. Skipping trial reports.")
        return

    df_ml = df_ml.copy()
    df_ml["timestamp"] = pd.to_datetime(df_ml["timestamp"])

    df_ml["trial_name"] = df_ml["trial_metadata"].apply(
        lambda x: x.get("name") if isinstance(x, dict) else "Unknown"
    )

    plant_status = df_ml.groupby("channel_id")["label"].max().rename("global_status")
    df_ml = df_ml.merge(plant_status, on="channel_id")

    for trial_name, group in df_ml.groupby("trial_name"):
        stress_events = group[group["label"] == 1]
        stress_start = stress_events["timestamp"].min() if not stress_events.empty else None

        df_ctrl = group[group["global_status"] == 0].set_index("timestamp")
        df_str = group[group["global_status"] == 1].set_index("timestamp")

        common_idx = group.set_index("timestamp").resample(resample_freq).mean(numeric_only=True).index
        df_plot = pd.DataFrame(index=common_idx)

        if not df_ctrl.empty:
            df_plot["prob_control"] = df_ctrl["prob_stress"].resample(resample_freq).mean()
        if not df_str.empty:
            df_plot["prob_stress"] = df_str["prob_stress"].resample(resample_freq).mean()

        df_plot = df_plot.interpolate(method='time', limit=2)

        out_path = os.path.join(trial_reports_dir, f"curve_{trial_name}.html")
        DownstreamTaskVisualizer.plot_stress_probability_dynamics(
            df_trial=df_plot,
            trial_name=str(trial_name),
            stress_start_time=stress_start,
            output_html_path=out_path
        )
        logger.info(f"Trial report saved -> trial_reports/curve_{trial_name}.html")


def main(data_source: Optional[DataSource] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--save_anchor", type=str, default=None, help="Path to save computed anchors")
    parser.add_argument("--load_anchor", type=str, default=None, help="Path to load existing anchors")
    parser.add_argument("--top_n_plants", type=int, default=3, help="Number of highly anomalous plants to visualize")
    parser.add_argument("--context_days", type=float, default=3.0, help="Days of context around the anomaly peak")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = parse_config(f)

    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    folder_name = f"{timestamp}_test_{config['experiment_name']}"
    output_dir = os.path.join("outputs", folder_name)

    model = build_model_from_config(config)
    model.load_state_dict(torch.load(config["model"]["init_weight"], map_location=device), strict=False)
    model.to(device)

    eval_type = config.get("evaluation", {}).get("type")
    if eval_type == "AnomalyEvaluation":
        evaluator = AnomalyEvaluation(device=device)
    elif eval_type == "SupervisedEvaluation":
        evaluator = SupervisedEvaluation(device=device)
    elif eval_type == "RepresentationEvaluation":
        evaluator = RepresentationEvaluation(device=device)
    else:
        raise ValueError(f"Type d'évaluation inconnu ou manquant : {eval_type}")

    anchor_streamer, test_streamer = get_dataloaders(
        config, logger, data_source, eval_split="test"
    )

    try:
        if hasattr(evaluator, "setup_anchors"):
            anchor_loader = stream_wrapper(anchor_streamer, device)
            evaluator.setup_anchors(model, anchor_loader=anchor_loader, load_path=args.load_anchor,
                                    save_path=args.save_anchor)
    finally:
        anchor_streamer.stop()

    logger.info("Running inference on test set...")
    test_loader = stream_wrapper(test_streamer, device)
    try:
        raw_results = evaluator.run_inference(model, test_loader)
    finally:
        test_streamer.stop()

    report = evaluator.compute_metrics(raw_results)

    logger.info("--- EVALUATION METRICS ---")
    for key, value in report.get("scalars", {}).items():
        if isinstance(value, (int, float)):
            logger.info(f"    {key:<25} : {value:.4f}")
        else:
            logger.info(f"    {key:<25} : {value}")

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(report.get("scalars", {}), f, indent=4)

    if "curves" in report:
        with open(os.path.join(output_dir, "roc_curve_data.json"), "w") as f:
            json.dump(report["curves"], f, indent=4)

        DownstreamTaskVisualizer.plot_roc_curve(
            curves_data=report["curves"],
            roc_auc=report.get("scalars", {}).get("roc_auc", 0.0),
            output_html_path=os.path.join(output_dir, "roc_curve.html")
        )
        logger.info("ROC curve saved -> roc_curve.html")

    metadata = report.get("metadata") or []
    df_ml = pd.DataFrame(metadata)

    outputs = raw_results["outputs"]
    if eval_type == "SupervisedEvaluation" and outputs.ndim > 1:
        df_ml["prob_stress"] = outputs[:, 1]
    else:
        df_ml["score"] = outputs.flatten()

    if raw_results["labels"] is not None:
        df_ml["label"] = raw_results["labels"].tolist()

    can_reconstruct_signal = raw_results.get("raw_windows") is not None
    if can_reconstruct_signal:
        df_ml["_raw_window"] = list(raw_results["raw_windows"])

    out_csv = os.path.join(output_dir, "ml_results_latest.csv")
    csv_cols = [c for c in df_ml.columns if c != "_raw_window"]
    df_ml[csv_cols].to_csv(out_csv, index=False)
    logger.info(f"Raw predictions saved -> {out_csv}")

    if args.top_n_plants > 0 and "score" in df_ml.columns:
        build_plant_reports(
            df_ml=df_ml,
            output_dir=output_dir,
            top_n_plants=args.top_n_plants,
            context_days=args.context_days,
            sampling_rate_sec=config["data"].get("sampling_rate_sec", 1.0),
            can_reconstruct_signal=can_reconstruct_signal
        )

    if "prob_stress" in df_ml.columns and "label" in df_ml.columns:
        build_trial_reports(
            df_ml=df_ml,
            output_dir=output_dir,
            resample_freq='6h'
        )

    logger.info(f"Evaluation complete. Results in: {output_dir}/")

if __name__ == "__main__":
    main()