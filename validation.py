import os
import argparse
import logging
import json
import torch
import pandas as pd
import numpy as np
from typing import Optional

from vivml.SSL.config_yml import parse_config
from vivml.SSL.model.builder import build_model_from_config, init_model_weights
from vivml.SSL.evaluation.builder import run_full_evaluation_pipeline
from vivml.SSL.data_streamer.wrapper import stream_wrapper
from vivml.SSL.data_streamer.builder import get_dataloaders
from vivml.SSL.data_streamer.source import DataSource
from vivml.SSL.evaluation.visualizer import LatentSpaceVisualizer, DownstreamTaskVisualizer

logger = logging.getLogger(__name__)


def extract_sample_latents(model, streamer, device, max_batches=150):
    loader = stream_wrapper(streamer, device)
    model.eval()
    all_outputs, all_labels, all_metadata = [], [], []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            if not batch or len(batch) == 0:
                continue

            if len(batch) == 3:
                x_raw, y_raw, meta = batch
            else:
                x_raw, meta = batch[0], batch[1]
                y_raw = None

            x = torch.tensor(x_raw, dtype=torch.float32).to(device) if not isinstance(x_raw, torch.Tensor) else x_raw.to(device)
            if x.dim() == 2:
                x = x.unsqueeze(1)

            latents = model.get_projection(x, stats=meta).cpu().numpy()
            all_outputs.append(latents)

            if y_raw is not None:
                labels = y_raw.cpu().numpy() if isinstance(y_raw, torch.Tensor) else y_raw
                all_labels.extend(labels)

            batch_size = x.size(0)
            if meta is not None:
                if isinstance(meta, dict):
                    for b in range(batch_size):
                        item = {}
                        for k, v in meta.items():
                            if isinstance(v, (list, tuple)) and len(v) == batch_size:
                                item[k] = v[b]
                            elif isinstance(v, torch.Tensor) and v.size(0) == batch_size:
                                item[k] = v[b].item() if v.dim() == 1 else v[b].tolist()
                            else:
                                item[k] = v
                        all_metadata.append(item)
                elif isinstance(meta, list) and len(meta) == batch_size:
                    all_metadata.extend(meta)
                else:
                    all_metadata.extend([{}] * batch_size)
            else:
                all_metadata.extend([{}] * batch_size)

    if all_outputs:
        return np.vstack(all_outputs), np.array(all_labels), all_metadata
    return np.array([]), np.array([]), []


def evaluate_model(
    config,
    run_dir,
    model_method=None,
    data_source: Optional[DataSource] = None,
):
    device = torch.device(config.get("device", "cpu") if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting evaluation for experiment: {config.get('experiment_name')}")

    if model_method is None:
        weights_path = os.path.join(run_dir, "full_pretext_model.pt")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Missing model weights at {weights_path}")

        model_method = build_model_from_config(config)
        init_model_weights(model_method, config, explicit_weight_path=weights_path)

    model_method.to(device)

    train_streamer, val_streamer = get_dataloaders(
        config,
        logger,
        data_source,
        val_max_plant_days_override=1500
    )

    try:
        return _evaluate_streamers(
            config, run_dir, model_method, device, train_streamer, val_streamer
        )
    finally:
        try:
            train_streamer.stop()
        finally:
            val_streamer.stop()


def _evaluate_streamers(config, run_dir, model_method, device, train_streamer, val_streamer):
    eval_type = config.get("evaluation", {}).get("type")
    if eval_type != "SupervisedEvaluation":
        logger.info("Stopping unused train_streamer to free RAM.")
        train_streamer.stop()

    val_loader = stream_wrapper(val_streamer, device)

    metrics = run_full_evaluation_pipeline(
        config=config,
        model=model_method,
        val_loader=val_loader,
        device=device,
        run_dir=run_dir
    )

    logger.info("--- FINAL METRICS ---")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"    {key.capitalize():<20} : {value:.4f}")

    metrics_export = {k: v for k, v in metrics.items() if isinstance(v, float)}
    with open(os.path.join(run_dir, "final_evaluation_metrics.json"), "w") as f:
        json.dump(metrics_export, f, indent=4)

    eval_type = config.get("evaluation", {}).get("type")

    if eval_type == "SupervisedEvaluation":
        logger.info("Extracting Train set latents for domain shift analysis...")
        train_embeds, train_labels, train_meta = extract_sample_latents(
            model_method, train_streamer, device, max_batches=150
        )

        if "clusters" in metrics:
            clusters_data = metrics["clusters"]
            val_embeds = clusters_data["embeddings_np"]
            val_labels = (
                clusters_data.get("true_labels")
                if clusters_data.get("true_labels") is not None
                else clusters_data.get("cluster_labels")
            )
            val_meta = metrics.get("metadata", [])
        else:
            val_embeds, val_labels, val_meta = extract_sample_latents(
                model_method, val_streamer, device, max_batches=150
            )

        max_points_per_domain = 10000
        if len(train_embeds) > max_points_per_domain:
            indices = np.sort(np.random.choice(len(train_embeds), max_points_per_domain, replace=False))
            train_embeds = train_embeds[indices]
            train_labels = train_labels[indices] if train_labels is not None else None
            train_meta = [train_meta[i] for i in indices]

        if len(val_embeds) > max_points_per_domain:
            indices = np.sort(np.random.choice(len(val_embeds), max_points_per_domain, replace=False))
            val_embeds = val_embeds[indices]
            val_labels = val_labels[indices] if val_labels is not None else None
            val_meta = [val_meta[i] for i in indices]

        joint_embeds = (
            np.vstack([train_embeds, val_embeds])
            if train_embeds.size > 0
            else val_embeds
        )

        joint_labels = (
            np.concatenate([train_labels, val_labels])
            if train_labels is not None and val_labels is not None
            else None
        )

        if joint_labels is not None:
            try:
                joint_labels = np.round(np.array(joint_labels, dtype=float)).astype(int)
            except Exception:
                pass

        joint_metadata = []
        for m in train_meta:
            m_fixed = dict(m) if isinstance(m, dict) else {}
            trial_meta = m_fixed.get("trial_metadata", {})
            trial_meta = dict(trial_meta) if isinstance(trial_meta, dict) else {}
            trial_meta["Domain"] = "Train"
            m_fixed["trial_metadata"] = trial_meta
            joint_metadata.append(m_fixed)

        for m in val_meta:
            m_fixed = dict(m) if isinstance(m, dict) else {}
            trial_meta = m_fixed.get("trial_metadata", {})
            trial_meta = dict(trial_meta) if isinstance(trial_meta, dict) else {}
            trial_meta["Domain"] = "Val"
            m_fixed["trial_metadata"] = trial_meta
            joint_metadata.append(m_fixed)

        try:
            joint_domain = np.array(
                ["Train"] * len(train_embeds) + ["Val"] * len(val_embeds)
            )
            df_dict = {"domain": joint_domain}
            if joint_labels is not None and len(joint_labels) == len(joint_domain):
                df_dict["label"] = joint_labels

            df_export = pd.DataFrame(df_dict)

            latent_cols = [f"dim_{i}" for i in range(joint_embeds.shape[1])]
            latent_df = pd.DataFrame(joint_embeds, columns=latent_cols)
            df_export = pd.concat([df_export, latent_df], axis=1)

            export_path = os.path.join(run_dir, "latent_space_embeddings_joint.csv")
            df_export.to_csv(export_path, index=False)
        except Exception as e:
            logger.error(f"Failed to export joint embeddings CSV: {e}")

        fig = LatentSpaceVisualizer.plot_umap_clusters(
            embeddings_np=joint_embeds,
            cluster_labels=joint_labels,
            metadata=joint_metadata,
            title=f"Joint Latent Space (Train vs Val) - {config.get('experiment_name')}",
        )
        fig.write_html(os.path.join(run_dir, "joint_latent_space_umap.html"))

    elif "clusters" in metrics:
        clusters_data = metrics["clusters"]
        val_embeds = clusters_data["embeddings_np"]
        val_labels = (
            clusters_data.get("true_labels")
            if clusters_data.get("true_labels") is not None
            else clusters_data.get("cluster_labels")
        )
        val_meta = metrics.get("metadata", [])

        max_points_per_domain = 10000

        if len(val_embeds) > max_points_per_domain:
            indices = np.sort(np.random.choice(len(val_embeds), max_points_per_domain, replace=False))
            val_embeds = val_embeds[indices]
            val_labels = val_labels[indices] if val_labels is not None else None
            val_meta = [val_meta[i] for i in indices]

        fig = LatentSpaceVisualizer.plot_umap_clusters(
            embeddings_np=val_embeds,
            cluster_labels=val_labels,
            metadata=val_meta,
            title=f"Latent Space UMAP - {config.get('experiment_name')}",
        )
        fig.write_html(os.path.join(run_dir, "latent_space_umap.html"))

    if "curves" in metrics and "scalars" in metrics:
        curves_data = metrics["curves"]
        scalars = metrics["scalars"]
        cm_list = scalars.get("confusion_matrix", [[0, 0], [0, 0]])
        cm = np.array(cm_list)

        DownstreamTaskVisualizer.plot_roc_curve(
            curves_data=curves_data,
            roc_auc=scalars.get("roc_auc", 0.0),
            output_html_path=os.path.join(run_dir, "final_roc_curve.html")
        )
        DownstreamTaskVisualizer.plot_decision_thresholds(
            curves_data=curves_data,
            output_html_path=os.path.join(run_dir, "final_thresholds_analysis.html")
        )

        grid_key = "hourly_grid" if "hourly_grid" in metrics else "daily_grid"

        if grid_key in metrics and metrics[grid_key]:
            logger.info("Generation de la Heatmap d'optimisation...")

            DownstreamTaskVisualizer.plot_optimization_heatmap(
                grid_data=metrics[grid_key],
                output_html_path=os.path.join(run_dir, "optimization_heatmap.html"),
                window_label="1h" if grid_key == "hourly_grid" else "Day"
            )

            best_preds = metrics[grid_key].get("best_preds_continuous", [])
            best_labels = metrics[grid_key].get("best_labels", [])

            if best_preds and best_labels:
                logger.info("Generation du graphique des seuils pour le percentile optimal...")

                best_preds_np = np.array(best_preds, dtype=np.float32)
                best_labels_np = np.array(best_labels, dtype=np.int32)

                from sklearn.metrics import f1_score, confusion_matrix

                unique_thresholds = np.unique(best_preds_np)

                f1_scores, fps, fns = [], [], []

                for t in unique_thresholds:
                    preds = (best_preds_np >= t).astype(int)
                    f1_scores.append(float(f1_score(best_labels_np, preds, zero_division=0)))
                    cm = confusion_matrix(best_labels_np, preds, labels=[0, 1])
                    fps.append(int(cm[0, 1]) if cm.size == 4 else 0)
                    fns.append(int(cm[1, 0]) if cm.size == 4 else 0)

                optimal_curves = {
                    "thresholds": unique_thresholds.tolist(),
                    "f1_scores": f1_scores,
                    "false_positives": fps,
                    "false_negatives": fns,
                    "total_positives": int(np.sum(best_labels_np))
                }

                DownstreamTaskVisualizer.plot_decision_thresholds(
                    curves_data=optimal_curves,
                    output_html_path=os.path.join(run_dir, "optimal_percentile_thresholds.html")
                )

        DownstreamTaskVisualizer.plot_confusion_matrix(
            conf_matrix=cm,
            class_names=["Control", "Stress"],
            title="Confusion matrix (Threshold 0.5)",
            output_html_path=os.path.join(run_dir, "confusion_matrix.html")
        )

    return metrics


def main(data_source: Optional[DataSource] = None):
    parser = argparse.ArgumentParser(description="Agnostic evaluation runner")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to the training run output directory")
    args = parser.parse_args()

    run_dir = args.run_dir
    config_path = os.path.join(run_dir, "config_run.yaml")

    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found in {run_dir}.")
        return

    with open(config_path, "r") as f:
        config = parse_config(f)

    evaluate_model(
        config=config,
        run_dir=run_dir,
        model_method=None,
        data_source=data_source,
    )


if __name__ == "__main__":
    main()