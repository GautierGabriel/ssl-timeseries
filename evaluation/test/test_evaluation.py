import torch
import numpy as np
from unittest.mock import MagicMock, patch

from vivml.SSL.evaluation.evaluation import SupervisedEvaluation, RepresentationEvaluation, AnomalyEvaluation

class TestEvaluationEngines:

    def test__representation_evaluation__run_inference__loops_loader_and_accumulates(self):
        # GIVEN
        device = torch.device("cpu")
        engine = RepresentationEvaluation(device=device)

        mock_model = MagicMock()
        mock_model.get_projection.return_value = torch.rand(2, 8)
        # Mock de get_prediction pour éviter les erreurs
        mock_model.get_prediction.return_value = torch.rand(2, 2)

        mock_loader = [
            (torch.rand(2, 1, 100), torch.tensor([0, 1]), {"channel": ["A", "B"]}),
            (torch.rand(2, 1, 100), torch.tensor([1, 0]), {"channel": ["C", "D"]})
        ]

        # WHEN
        results = engine.run_inference(mock_model, mock_loader)

        # THEN
        assert "embeddings" in results
        assert "predictions" in results
        assert "labels" in results
        assert "metadata" in results

        assert results["embeddings"].shape == (4, 8)
        assert results["labels"].shape == (4,)
        assert len(results["metadata"]) == 4

    def test__supervised_evaluation__compute_metrics__calls_metrics_toolkit(self):
        # GIVEN
        engine = SupervisedEvaluation(device=torch.device("cpu"))

        raw_results = {
            "predictions": np.array([[0.9, 0.1], [0.1, 0.9]]),
            "embeddings": np.random.randn(2, 8),
            "labels": np.array([0, 1]),
            "metadata": [{"id": 1}, {"id": 2}]
        }

        # WHEN
        report = engine.compute_metrics(raw_results)

        # THEN
        assert "scalars" in report
        assert "accuracy" in report["scalars"]
        assert report["metadata"] == raw_results["metadata"]

    def test__representation_evaluation__compute_metrics__calls_clustering_toolkit(self):
        # GIVEN
        engine = RepresentationEvaluation(device=torch.device("cpu"), n_clusters=2)
        raw_results = {
            "embeddings": np.random.randn(4, 8),
            "labels": np.array([0, 0, 1, 1]),
            "metadata": [{"id": i} for i in range(4)]
        }

        # WHEN
        report = engine.compute_metrics(raw_results)

        # THEN
        assert "scalars" in report
        assert "silhouette_score" in report["scalars"]
        assert "clusters" in report
        assert report["clusters"]["embeddings_np"].shape == (4, 8)

class TestAnomalyEvaluation:

    def test__forward_pass__calls_model_directly_for_scores(self):
        # GIVEN
        device = torch.device("cpu")
        engine = AnomalyEvaluation(device=device)

        mock_model = MagicMock()
        expected_scores = torch.tensor([0.1, 0.8])
        mock_model.return_value = expected_scores

        x = torch.rand(2, 1, 100)
        meta = {"info": "test"}

        # WHEN
        out = engine._forward_pass(mock_model, x, meta)

        # THEN
        mock_model.assert_called_once_with(x, stats=meta)
        assert torch.equal(out, expected_scores)

    @patch("torch.load")
    def test__setup_anchors__loads_from_path(self, mock_torch_load):
        # GIVEN
        device = torch.device("cpu")
        engine = AnomalyEvaluation(device=device)

        mock_model = MagicMock()
        mock_loaded_tensor = torch.rand(1, 256)
        mock_torch_load.return_value = mock_loaded_tensor

        # WHEN
        engine.setup_anchors(mock_model, load_path="dummy/path.pt")

        # THEN
        mock_torch_load.assert_called_once_with("dummy/path.pt", map_location=device)
        mock_model.head.fit_anchors.assert_called_once_with(mock_loaded_tensor)

    def test__setup_anchors__builds_from_loader(self):
        # GIVEN
        device = torch.device("cpu")
        engine = AnomalyEvaluation(device=device)

        mock_model = MagicMock()
        mock_model.get_projection.side_effect = lambda x, stats: torch.rand(2, 256)

        mock_loader = [
            (torch.rand(2, 6, 100), {"meta": 1}),
            (torch.rand(2, 6, 100), {"meta": 2})
        ]

        # WHEN
        engine.setup_anchors(mock_model, anchor_loader=mock_loader)

        # THEN
        assert mock_model.get_projection.call_count == 2

        mock_model.head.fit_anchors.assert_called_once()
        args, _ = mock_model.head.fit_anchors.call_args
        assert args[0].shape == (4, 256)

    def test__compute_metrics__calculates_mean_distances_by_label(self):
        # GIVEN
        engine = AnomalyEvaluation(device=torch.device("cpu"))

        raw_results = {
            "outputs": np.array([0.1, 0.2, 0.8, 0.9]),
            "labels": np.array([0, 0, 1, 1]),
            "metadata": [{"id": i} for i in range(4)]
        }

        # WHEN
        report = engine.compute_metrics(raw_results)

        # THEN
        assert "scalars" in report
        scalars = report["scalars"]

        assert np.isclose(scalars["mean_distance_all"], 0.5)
        assert np.isclose(scalars["mean_distance_healthy_0"], 0.15)
        assert np.isclose(scalars["mean_distance_stress_1"], 0.85)

        assert report["raw_scores"] == [0.1, 0.2, 0.8, 0.9]
        assert report["metadata"] == raw_results["metadata"]

    def test__compute_metrics__handles_missing_labels_gracefully(self):
        # GIVEN
        engine = AnomalyEvaluation(device=torch.device("cpu"))
        raw_results = {
            "outputs": np.array([0.4, 0.6]),
            "labels": None,
            "metadata": []
        }

        # WHEN
        report = engine.compute_metrics(raw_results)

        # THEN
        assert "mean_distance_all" in report["scalars"]
        assert np.isclose(report["scalars"]["mean_distance_all"], 0.5)
        assert "mean_distance_healthy_0" not in report["scalars"]
        assert "mean_distance_stress_1" not in report["scalars"]