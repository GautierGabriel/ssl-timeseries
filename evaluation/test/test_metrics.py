import pytest
import torch
import numpy as np
from vivml.SSL.evaluation.metrics import ClusteringMetrics, ClassificationMetrics, AnomalyMetrics, HourlyStressMetrics, DailyStressMetrics


class TestClusteringMetrics:
    def test__compute_geometry__returns_expected_values(self):
        z1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        z2 = z1.clone()
        z_all = torch.cat([z1, z2], dim=0)

        metrics = ClusteringMetrics.compute_geometry(z1, z2, z_all)

        assert metrics["alignment"] == pytest.approx(0.0, abs=1e-5)
        assert isinstance(metrics["uniformity"], float)

    def test__compute_clustering_score__handles_small_input(self):
        embeddings = np.random.rand(3, 16)
        score = ClusteringMetrics.compute_clustering_score(embeddings, n_clusters=4)

        assert score["silhouette_score"] == 0.0
        assert len(score["cluster_labels"]) == 3


class TestClassificationMetrics:
    def test__compute_binary_stats(self):
        probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
        labels = np.array([0, 1, 1, 0])

        stats = ClassificationMetrics.compute_binary_stats(probs, labels)

        assert stats["true_positives"] == 1
        assert stats["false_positives"] == 1
        assert "f1_score" in stats
        assert 0.0 <= stats["accuracy"] <= 1.0

    def test__compute_threshold_curves(self):
        probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
        labels = np.array([0, 1, 1, 0])

        curves = ClassificationMetrics.compute_threshold_curves(probs, labels)

        assert len(curves["f1_scores"]) == 99
        assert "fpr" in curves and "tpr" in curves


class TestAnomalyMetrics:
    def test__compute_distance_stats__without_labels(self):
        distances = np.array([0.1, 0.2, 0.3, 0.4])

        stats = AnomalyMetrics.compute_distance_stats(distances)

        assert "mean_distance_all" in stats
        assert "std_distance_all" in stats
        assert stats["mean_distance_all"] == pytest.approx(0.25)

        assert "mean_distance_healthy_0" not in stats
        assert "roc_auc" not in stats

    def test__compute_distance_stats__with_labels_and_auc(self):
        distances = np.array([0.9, 0.8, 0.2, 0.1])
        labels = np.array([0, 0, 1, 1])

        stats = AnomalyMetrics.compute_distance_stats(distances, labels)

        assert "mean_distance_healthy_0" in stats
        assert "std_distance_healthy_0" in stats
        assert stats["mean_distance_healthy_0"] == pytest.approx(0.85)

        assert "mean_distance_stress_1" in stats
        assert "std_distance_stress_1" in stats
        assert stats["mean_distance_stress_1"] == pytest.approx(0.15)

        assert "roc_auc" in stats
        assert stats["roc_auc"] == 1.0

    def test__compute_distance_stats__with_single_class_graceful_fallback(self):
        distances = np.array([0.1, 0.2, 0.3])
        labels = np.array([0, 0, 0])

        stats = AnomalyMetrics.compute_distance_stats(distances, labels)

        assert "mean_distance_healthy_0" in stats
        assert "mean_distance_stress_1" not in stats

        assert "roc_auc" not in stats


class TestDailyStressMetrics:

    def test__evaluate_grid__computes_matrices_and_best_params(self):
        # GIVEN
        all_days_scores = [np.array([0.1, 0.2, 0.8, 0.9]), np.array([0.1, 0.1, 0.2, 0.1])]
        all_days_labels = [1, 0]
        percentiles = np.array([50, 90])
        thresholds = np.array([0.5, 0.8])

        # WHEN
        results = DailyStressMetrics.evaluate_grid(
            all_days_scores, all_days_labels, window_size=2,
            percentiles=percentiles, thresholds=thresholds
        )

        # THEN
        assert "f1_matrix" in results
        assert "best_params" in results
        assert np.array(results["f1_matrix"]).shape == (2, 2)
        assert results["best_params"]["f1_score"] > 0


class TestHourlyStressMetrics:

    def test__evaluate_grid__computes_hourly_predictions_and_matrices(self):
        # GIVEN
        all_days_scores = [np.array([0.1, 0.2, 0.8, 0.9]), np.array([0.1, 0.1, 0.2, 0.1])]
        all_days_labels = [1, 0]
        percentiles = np.array([50, 90])
        thresholds = np.array([0.5, 0.8])

        # WHEN
        results = HourlyStressMetrics.evaluate_grid(
            all_days_scores, all_days_labels, window_size=2,
            percentiles=percentiles, thresholds=thresholds
        )

        # THEN
        assert "best_preds_continuous" in results
        assert "best_labels" in results
        assert len(results["best_labels"]) == 4
        assert np.array(results["f1_matrix"]).shape == (2, 2)