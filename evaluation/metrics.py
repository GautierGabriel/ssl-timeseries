import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, silhouette_score


class ClusteringMetrics:
    @staticmethod
    def compute_geometry(z1: torch.Tensor, z2: torch.Tensor, z_all: torch.Tensor) -> Dict[str, float]:
        p1 = F.normalize(z1, p=2, dim=1)
        p2 = F.normalize(z2, p=2, dim=1)
        alignment = torch.pow(torch.norm(p1 - p2, p=2, dim=1), 2).mean().item()

        max_geometry_samples = 15000
        if z_all.size(0) > max_geometry_samples:
            indices = torch.randperm(z_all.size(0), device=z_all.device)[:max_geometry_samples]
            z_all = z_all[indices]

        p_all = F.normalize(z_all, p=2, dim=1)
        sq_dist = torch.cdist(p_all, p_all, p=2).pow(2)
        mask = torch.eye(p_all.size(0), dtype=torch.bool, device=z_all.device)
        sq_dist = sq_dist.masked_select(~mask)
        uniformity = torch.log(torch.exp(-2.0 * sq_dist).mean()).item()

        return {"alignment": alignment, "uniformity": uniformity}

    @staticmethod
    def compute_clustering_score(embeddings_np: np.ndarray, n_clusters: int = 4) -> Dict[str, Any]:
        n_samples = embeddings_np.shape[0]
        if n_samples <= n_clusters:
            return {"silhouette_score": 0.0, "cluster_labels": np.zeros(n_samples)}

        max_samples = 15000
        if n_samples > max_samples:
            indices = np.random.choice(n_samples, max_samples, replace=False)
            train_embeds = embeddings_np[indices]
        else:
            train_embeds = embeddings_np

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        kmeans.fit(train_embeds)

        cluster_labels = kmeans.predict(embeddings_np)

        train_labels = cluster_labels[indices] if n_samples > max_samples else cluster_labels
        max_silhouette_samples = 3000

        silhouette = float(silhouette_score(
            train_embeds,
            train_labels,
            sample_size=min(max_silhouette_samples, train_embeds.shape[0]),
            random_state=42
        ))

        return {"silhouette_score": silhouette, "cluster_labels": cluster_labels}

class ClassificationMetrics:
    @staticmethod
    def compute_binary_stats(probs: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        labels = np.asarray(labels).ravel()

        pos_probs = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
        preds = (pos_probs >= 0.5).astype(int)
        preds = preds.ravel()

        if labels.shape[0] != preds.shape[0]:
            raise ValueError(f"Shape mismatch: labels has {labels.shape[0]} elements, preds has {preds.shape[0]} elements.")

        cm = confusion_matrix(labels, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "f1_score": float(f1_score(labels, preds, average='macro')),
            "roc_auc": float(roc_auc_score(labels, pos_probs)),
            "precision": precision,
            "recall_sensitivity": recall,
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "confusion_matrix": cm.tolist()
        }

    @staticmethod
    def compute_threshold_curves(probs: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        pos_probs = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
        thresholds = np.linspace(0.01, 0.99, 99)

        f1_curve, fn_curve, fp_curve = [], [], []

        for t in thresholds:
            preds_t = (pos_probs >= t).astype(int)
            f1_curve.append(float(f1_score(labels, preds_t, average='macro')))

            cm_t = confusion_matrix(labels, preds_t, labels=[0, 1])
            _, fp_t, fn_t, _ = cm_t.ravel() if cm_t.size == 4 else (0, 0, 0, 0)

            fn_curve.append(int(fn_t))
            fp_curve.append(int(fp_t))

        fpr, tpr, _ = roc_curve(labels, pos_probs)

        return {
            "thresholds": thresholds.tolist(),
            "f1_scores": f1_curve,
            "false_negatives": fn_curve,
            "false_positives": fp_curve,
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "total_positives": int(np.sum(labels == 1))
        }


class AnomalyMetrics:
    @staticmethod
    def compute_distance_stats(distances: np.ndarray, labels: np.ndarray = None) -> Dict[str, float]:
        stats = {
            "mean_distance_all": float(distances.mean()),
            "std_distance_all": float(distances.std())
        }

        if labels is not None:
            mask_healthy = (labels == 0)
            mask_stress = (labels == 1)

            if mask_healthy.any():
                stats["mean_distance_healthy_0"] = float(distances[mask_healthy].mean())
                stats["std_distance_healthy_0"] = float(distances[mask_healthy].std())

            if mask_stress.any():
                stats["mean_distance_stress_1"] = float(distances[mask_stress].mean())
                stats["std_distance_stress_1"] = float(distances[mask_stress].std())

            if mask_healthy.any() and mask_stress.any():
                from sklearn.metrics import roc_auc_score
                stats["roc_auc"] = float(roc_auc_score(labels, -distances))

        return stats

    @staticmethod
    def compute_roc_curve_data(distances: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        fpr, tpr, thresholds = roc_curve(labels, distances)
        return {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist()
        }


class DailyStressMetrics:
    @staticmethod
    def evaluate_grid(
            all_days_scores: List[np.ndarray],
            all_days_labels: List[int],
            window_size: int = 6,
            percentiles: np.ndarray = np.arange(50, 100, 2),
            thresholds: np.ndarray = np.linspace(0.01, 0.99, 50)
    ) -> Dict[str, Any]:
        labels = np.array(all_days_labels)

        f1_matrix = np.zeros((len(percentiles), len(thresholds)))
        tp_matrix = np.zeros((len(percentiles), len(thresholds)))
        tn_matrix = np.zeros((len(percentiles), len(thresholds)))
        fp_matrix = np.zeros((len(percentiles), len(thresholds)))
        fn_matrix = np.zeros((len(percentiles), len(thresholds)))

        best_f1 = 0.0
        best_params = {}

        for i, p in enumerate(percentiles):
            daily_preds_p = []

            for day_scores in all_days_scores:
                if len(day_scores) < window_size:
                    val = float(np.percentile(day_scores, p)) if len(day_scores) > 0 else 0.0
                else:
                    macro_windows = np.lib.stride_tricks.sliding_window_view(day_scores, window_shape=window_size)
                    window_densities = np.percentile(macro_windows, p, axis=1)
                    val = float(np.max(window_densities))

                daily_preds_p.append(val)

            daily_preds_p = np.array(daily_preds_p)

            for j, t in enumerate(thresholds):
                preds = (daily_preds_p >= t).astype(int)
                f1 = float(f1_score(labels, preds, zero_division=0))

                cm = confusion_matrix(labels, preds, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

                f1_matrix[i, j] = f1
                tp_matrix[i, j] = int(tp)
                tn_matrix[i, j] = int(tn)
                fp_matrix[i, j] = int(fp)
                fn_matrix[i, j] = int(fn)

                if f1 > best_f1:
                    best_f1 = f1
                    best_params = {
                        'percentile': float(p),
                        'threshold': float(t),
                        'f1_score': f1,
                        'tp': int(tp), 'tn': int(tn),
                        'fp': int(fp), 'fn': int(fn)
                    }

        return {
            "f1_matrix": f1_matrix.tolist(),
            "tp_matrix": tp_matrix.tolist(),
            "tn_matrix": tn_matrix.tolist(),
            "fp_matrix": fp_matrix.tolist(),
            "fn_matrix": fn_matrix.tolist(),
            "percentiles": percentiles.tolist(),
            "thresholds": thresholds.tolist(),
            "best_params": best_params
        }


class HourlyStressMetrics:
    @staticmethod
    def evaluate_grid(
            all_days_scores: List[np.ndarray],
            all_days_labels: List[int],
            window_size: int = 6,
            percentiles: np.ndarray = np.arange(50, 100, 2),
            thresholds: np.ndarray = np.linspace(0.01, 0.30, 40)
    ) -> Dict[str, Any]:

        f1_matrix = np.zeros((len(percentiles), len(thresholds)))
        tp_matrix = np.zeros((len(percentiles), len(thresholds)))
        tn_matrix = np.zeros((len(percentiles), len(thresholds)))
        fp_matrix = np.zeros((len(percentiles), len(thresholds)))
        fn_matrix = np.zeros((len(percentiles), len(thresholds)))

        best_f1 = 0.0
        best_params = {}
        best_preds_continuous = []
        best_labels_out = []

        for i, p in enumerate(percentiles):
            hourly_preds = []
            hourly_labels = []

            for day_scores, day_label in zip(all_days_scores, all_days_labels):
                num_blocks = len(day_scores) // window_size

                if num_blocks == 0:
                    if len(day_scores) > 0:
                        hourly_preds.append(float(np.percentile(day_scores, p)))
                        hourly_labels.append(day_label)
                    continue

                blocks = day_scores[:num_blocks * window_size].reshape(num_blocks, window_size)
                block_densities = np.percentile(blocks, p, axis=1)

                hourly_preds.extend(block_densities.tolist())
                hourly_labels.extend([day_label] * num_blocks)

            hourly_preds = np.array(hourly_preds)
            hourly_labels = np.array(hourly_labels)

            for j, t in enumerate(thresholds):
                preds = (hourly_preds >= t).astype(int)
                f1 = float(f1_score(hourly_labels, preds, zero_division=0))

                cm = confusion_matrix(hourly_labels, preds, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

                f1_matrix[i, j] = f1
                tp_matrix[i, j] = int(tp)
                tn_matrix[i, j] = int(tn)
                fp_matrix[i, j] = int(fp)
                fn_matrix[i, j] = int(fn)

                if f1 > best_f1:
                    best_f1 = f1
                    best_params = {
                        'percentile': float(p),
                        'threshold': float(t),
                        'f1_score': f1,
                        'tp': int(tp), 'tn': int(tn),
                        'fp': int(fp), 'fn': int(fn)
                    }
                    best_preds_continuous = hourly_preds
                    best_labels_out = hourly_labels

        return {
            "f1_matrix": f1_matrix.tolist(),
            "tp_matrix": tp_matrix.tolist(),
            "tn_matrix": tn_matrix.tolist(),
            "fp_matrix": fp_matrix.tolist(),
            "fn_matrix": fn_matrix.tolist(),
            "percentiles": percentiles.tolist(),
            "thresholds": thresholds.tolist(),
            "best_params": best_params,
            "best_preds_continuous": best_preds_continuous.tolist() if len(best_preds_continuous) > 0 else [],
            "best_labels": best_labels_out.tolist() if len(best_labels_out) > 0 else []
        }