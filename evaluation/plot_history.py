import os
import glob
import json
import argparse
import yaml
import torch
import numpy as np
import logging
import pandas as pd
import umap
import plotly.express as px
from scipy.linalg import orthogonal_procrustes

from vivml.SSL.evaluation.builder import build_evaluator_from_config
from vivml.SSL.evaluation.visualizer import LatentSpaceVisualizer, DownstreamTaskVisualizer

logger = logging.getLogger(__name__)


def process_history_and_plots(run_dir: str):
    history_dir = os.path.join(run_dir, "history")
    config_path = os.path.join(run_dir, "config_run.yaml")
    npz_files = sorted(glob.glob(os.path.join(history_dir, "*_latent.npz")))

    if not npz_files or not os.path.exists(config_path):
        logger.warning(f"Missing history files or config in {run_dir}")
        return

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cpu")
    engine = build_evaluator_from_config(config, device)

    history_json_data = []
    temporal_ssl_data = []

    for file_path in npz_files:
        epoch = int(os.path.basename(file_path).split("_")[1])
        data = np.load(file_path, allow_pickle=True)

        labels_data = data.get("labels")
        if labels_data is not None and labels_data.shape == () and labels_data[()] is None:
            labels_data = None

        metadata_data = data.get("metadata", None)
        if metadata_data is not None:
            if isinstance(metadata_data, np.ndarray):
                metadata_data = metadata_data.item() if metadata_data.ndim == 0 else metadata_data.tolist()

        raw_results = {
            "embeddings": data["embeddings"] if "embeddings" in data else None,
            "predictions": data["predictions"] if "predictions" in data else None,
            "labels": data["labels"] if "labels" in data else None,
            "metadata": data["metadata"].tolist() if "metadata" in data else None
        }

        report = engine.compute_metrics(raw_results)
        scalars = report.get("scalars", {})

        json_entry = {"epoch": epoch}
        json_entry.update(scalars)
        history_json_data.append(json_entry)

        if "curves" in report:
            curves = report["curves"]
            roc_auc = scalars.get("roc_auc", 0.0)

            DownstreamTaskVisualizer.plot_roc_curve(
                curves_data=curves,
                roc_auc=roc_auc,
                output_html_path=os.path.join(history_dir, f"roc_curve_epoch_{epoch:03d}.html")
            )
            DownstreamTaskVisualizer.plot_decision_thresholds(
                curves_data=curves,
                output_html_path=os.path.join(history_dir, f"thresholds_epoch_{epoch:03d}.html")
            )

        if "clusters" in report:
            clusters = report["clusters"]
            silhouette = scalars.get("silhouette_score", 0.0)
            plot_labels = clusters.get("cluster_labels", clusters.get("true_labels"))

            temporal_ssl_data.append({
                "epoch": epoch,
                "embeddings": clusters["embeddings_np"],
                "labels": plot_labels,
                "silhouette": silhouette,
                "metadata": metadata_data
            })

    with open(os.path.join(history_dir, "calculated_history_metrics.json"), "w") as f:
        json.dump(history_json_data, f, indent=4)

    if temporal_ssl_data:
        logger.info("Aligning latent spaces onto a global continuous UMAP referential...")
        temporal_ssl_data.sort(key=lambda x: x["epoch"])

        MAX_POINTS = 1500

        ref_frame = temporal_ssl_data[-1]
        n_points_total = len(ref_frame["embeddings"])

        if n_points_total > MAX_POINTS:
            logger.info(f"Subsampling history points from {n_points_total} to {MAX_POINTS} for fast visualization...")
            np.random.seed(42)
            sample_indices = np.random.choice(n_points_total, MAX_POINTS, replace=False)
        else:
            sample_indices = np.arange(n_points_total)

        logger.info("Applying Orthogonal Procrustes alignment to cancel latent space drift...")
        ref_emb = ref_frame["embeddings"][sample_indices]
        ref_centered = ref_emb - ref_emb.mean(axis=0)

        aligned_embeddings_list = []

        for frame in temporal_ssl_data:
            emb = frame["embeddings"][sample_indices]
            emb_centered = emb - emb.mean(axis=0)

            R, _ = orthogonal_procrustes(emb_centered, ref_centered)

            aligned_emb = emb_centered @ R
            aligned_embeddings_list.append(aligned_emb)

        all_embeddings = np.vstack(aligned_embeddings_list)

        logger.info(f"Fitting global UMAP on {len(all_embeddings)} aligned points...")
        reducer = umap.UMAP(n_components=2, random_state=42, n_jobs=1)
        all_emb_2d = reducer.fit_transform(all_embeddings)

        all_frames = []
        current_idx = 0

        logger.info("Generating HTML plots with metadata (fast mode)...")
        for i, frame in enumerate(temporal_ssl_data):
            epoch = frame["epoch"]
            current_len = len(sample_indices)

            start_idx = current_idx
            end_idx = current_idx + current_len
            emb_2d = all_emb_2d[start_idx:end_idx]
            current_idx = end_idx

            sampled_labels = frame["labels"][sample_indices]

            df_dict = {
                "x": emb_2d[:, 0],
                "y": emb_2d[:, 1],
                "Cluster": [str(lbl) for lbl in sampled_labels],
                "Epoch": epoch
            }

            if frame.get("metadata"):
                sampled_meta_full = [frame["metadata"][idx] for idx in sample_indices]
                for key in sampled_meta_full[0].keys():
                    df_dict[key] = [str(m.get(key, "N/A")) for m in sampled_meta_full]

            df_frame = pd.DataFrame(df_dict)
            all_frames.append(df_frame)

            hover_cols = [col for col in df_frame.columns if col not in ["x", "y", "Epoch"]]

            color_target = "channel_id" if "channel_id" in df_frame.columns else "Cluster"

            fig_static = px.scatter(
                df_frame,
                x="x", y="y",
                color=color_target,
                hover_data=hover_cols,
                title=f"Espace Latent - Epoque {epoch} (Silhouette: {frame['silhouette']:.3f})"
            )
            fig_static.write_html(os.path.join(history_dir, f"umap_epoch_{epoch:03d}.html"))

        logger.info("Building final animation slider...")
        df_temporal = pd.concat(all_frames, ignore_index=True)

        hover_cols_anim = [col for col in df_temporal.columns if col not in ["x", "y", "Epoch"]]
        color_target_anim = "channel_id" if "channel_id" in df_temporal.columns else "Cluster"

        fig_anim = px.scatter(
            df_temporal,
            x="x", y="y",
            color=color_target_anim,
            hover_data=hover_cols_anim,
            animation_frame="Epoch",
            title=f"Dynamique de l'espace latent (Coloré par: {color_target_anim})",
            labels={"x": "UMAP 1", "y": "UMAP 2"}
        )

        fig_anim.update_xaxes(range=[df_temporal["x"].min() - 1, df_temporal["x"].max() + 1])
        fig_anim.update_yaxes(range=[df_temporal["y"].min() - 1, df_temporal["y"].max() + 1])

        if fig_anim.layout.updatemenus:
            fig_anim.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 500

        fig_anim.write_html(os.path.join(history_dir, "temporal_latent_space_slider.html"))

        anim_path = os.path.join(history_dir, "temporal_latent_space_slider.html")
        fig_anim.write_html(anim_path)
        logger.info(f"Temporal slider saved successfully to {anim_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    args = parser.parse_args()
    process_history_and_plots(args.run_dir)