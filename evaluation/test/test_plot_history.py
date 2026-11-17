import os
import yaml
import numpy as np
from vivml.SSL.evaluation.plot_history import process_history_and_plots


class TestPlotHistory:

    def test_aborts_if_no_files(self, tmp_path):
        process_history_and_plots(str(tmp_path))
        assert not (tmp_path / "history" / "calculated_history_metrics.json").exists()

    def test_process_history_ssl_with_metadata(self, tmp_path):
        config = {"evaluation": {"type": "RepresentationEvaluation", "n_clusters": 3}}
        (tmp_path / "config_run.yaml").write_text(yaml.dump(config))

        history_dir = tmp_path / "history"
        history_dir.mkdir()

        mock_metadata = [{"channel_id": f"CH_{i % 4}", "device_type": "A"} for i in range(20)]

        for epoch in ["010", "020"]:
            np.savez(
                history_dir / f"epoch_{epoch}_latent.npz",
                embeddings=np.random.rand(20, 16),  # Correction: outputs -> embeddings
                labels=np.random.randint(0, 3, 20),
                metadata=mock_metadata
            )

        process_history_and_plots(str(tmp_path))

        assert (history_dir / "calculated_history_metrics.json").exists()
        assert (history_dir / "umap_epoch_010.html").exists()

    def test_process_history_supervised_without_metadata(self, tmp_path):
        config = {"evaluation": {"type": "SupervisedEvaluation"}}
        (tmp_path / "config_run.yaml").write_text(yaml.dump(config))

        history_dir = tmp_path / "history"
        history_dir.mkdir()

        mock_predictions = np.random.rand(20, 2)
        mock_predictions = mock_predictions / mock_predictions.sum(axis=1, keepdims=True)
        mock_labels = np.random.randint(0, 2, 20)
        mock_labels[0], mock_labels[1] = 0, 1

        np.savez(
            history_dir / "epoch_010_latent.npz",
            predictions=mock_predictions,  # Correction: outputs -> predictions
            embeddings=np.random.rand(20, 8),  # SupervisedEvaluation attend des embeddings
            labels=mock_labels
        )

        process_history_and_plots(str(tmp_path))

        assert (history_dir / "calculated_history_metrics.json").exists()
        assert (history_dir / "roc_curve_epoch_010.html").exists()