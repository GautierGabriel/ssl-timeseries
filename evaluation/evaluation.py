import logging
import os
import torch
from .baseevaluation import BaseEvaluationEngine
import pandas as pd
import numpy as np
from vivml.SSL.evaluation.metrics import ClassificationMetrics, ClusteringMetrics, AnomalyMetrics, HourlyStressMetrics

logger = logging.getLogger(__name__)


class SupervisedEvaluation(BaseEvaluationEngine):

    def compute_metrics(self, raw_results: dict) -> dict:
        logger.info("Starting supervised evaluation metrics computation.")

        # Ensure embeddings are a NumPy array
        embeddings = raw_results["embeddings"]
        if hasattr(embeddings, "shape") and not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)

        # Extract predictions unconditionally
        probs = raw_results.get("predictions")

        # Ensure probs are a NumPy array if they exist
        if probs is not None and hasattr(probs, "shape") and not isinstance(probs, np.ndarray):
            probs = np.array(probs)

        if probs is None:
            raise ValueError("No predictions found in raw_results. Supervised evaluation requires predictions.")

        labels = raw_results.get("labels")
        if labels is None:
            raise ValueError("No labels found in raw_results. Supervised evaluation requires labels.")

        stats = ClassificationMetrics.compute_binary_stats(probs, labels)
        curves = ClassificationMetrics.compute_threshold_curves(probs, labels)

        pos_probs = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
        metadata = raw_results["metadata"]

        daily_data = {}

        if metadata is not None:
            for idx, meta in enumerate(metadata):
                ts = pd.to_datetime(meta.get("timestamp"))
                day_str = ts.strftime('%Y-%m-%d') if pd.notnull(ts) else "unknown_day"
                plant_id = meta.get("channel_id", "unknown_plant")

                key = f"{plant_id}_{day_str}"
                if key not in daily_data:
                    daily_data[key] = {"scores": [], "label": labels[idx]}
                daily_data[key]["scores"].append(pos_probs[idx])
            pass

        all_days_scores = [np.array(d["scores"]) for d in daily_data.values()]
        all_days_labels = [d["label"] for d in daily_data.values()]

        daily_grid = {}
        if all_days_scores:
            daily_grid = HourlyStressMetrics.evaluate_grid(all_days_scores, all_days_labels)

        return {
            "scalars": stats,
            "curves": curves,
            "daily_grid": daily_grid,
            "metadata": raw_results["metadata"],
            "clusters": {
                "embeddings_np": embeddings,
                "true_labels": labels
            }
        }


class RepresentationEvaluation(BaseEvaluationEngine):

    def compute_metrics(self, raw_results: dict) -> dict:
        logger.info("Starting representation evaluation metrics computation.")
        embeddings = raw_results["embeddings"]

        logger.debug(f"RepEval - Embeddings received: Shape = {embeddings.shape}, dtype = {embeddings.dtype}")

        if embeddings.ndim == 4:
            logger.warning("Fallback: 4D to 2D global pooling triggered.")
            embeddings = embeddings.mean(axis=(1, 2))
        elif embeddings.ndim == 3:
            logger.warning("Fallback: 3D to 2D global pooling triggered.")
            embeddings = embeddings.mean(axis=1)

        labels = raw_results.get("labels")
        n_clusters = getattr(self, "n_clusters", 4)

        logger.debug(f"Computing clustering score ({n_clusters} clusters)...")
        cluster_stats = ClusteringMetrics.compute_clustering_score(embeddings, n_clusters=n_clusters)

        return {
            "scalars": {
                "silhouette_score": cluster_stats["silhouette_score"]
            },
            "clusters": {
                "embeddings_np": embeddings,
                "cluster_labels": cluster_stats["cluster_labels"],
                "true_labels": labels
            },
            "metadata": raw_results["metadata"]
        }


class AnomalyEvaluation(BaseEvaluationEngine):

    def _forward_pass(self, model, x, meta):
        return model(x, stats=meta)

    def setup_anchors(self, model, anchor_loader=None, load_path=None, save_path=None):
        if load_path:
            logger.info(f"Loading anchor from {load_path}")
            model.head.fit_anchors(torch.load(load_path, map_location=self.device))
            return

        if anchor_loader is None:
            raise ValueError("anchor_loader must be provided to build a new anchor.")

        logger.info("Building new anchor from 'train' dataloader...")
        all_latents = []

        with torch.no_grad():
            for batch in anchor_loader:
                if len(batch) == 3:
                    x, _, meta = batch
                else:
                    x, meta = batch[0], batch[1]

                latent = model.get_projection(x.to(self.device), stats=meta)
                all_latents.append(latent)

        global_latent_tensor = torch.cat(all_latents, dim=0)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(global_latent_tensor, save_path)
            logger.info(f"Anchor saved to {save_path}")

        model.head.fit_anchors(global_latent_tensor)

    def compute_metrics(self, raw_results: dict) -> dict:
        scores = raw_results["outputs"].flatten()
        labels = raw_results["labels"]
        stats = AnomalyMetrics.compute_distance_stats(distances=scores, labels=labels)

        report = {
            "scalars": stats,
            "metadata": raw_results["metadata"],
            "raw_scores": scores.tolist()
        }

        if labels is not None:
            report["curves"] = AnomalyMetrics.compute_roc_curve_data(scores, labels)

        return report