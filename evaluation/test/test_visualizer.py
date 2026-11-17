import json
import pytest
import numpy as np
import plotly.graph_objects as go
import pandas as pd

from vivml.SSL.evaluation.visualizer import TrainingVisualizer, LatentSpaceVisualizer, AnomalyVisualizer, DownstreamTaskVisualizer

class TestLatentSpaceVisualizer:

    def test__plot_umap_clusters__returns_valid_plotly_figure_without_metadata(self):
        # GIVEN :
        np.random.seed(42)

        embeddings_np = np.random.rand(20, 16)
        cluster_labels = np.array([0] * 7 + [1] * 7 + [2] * 6)
        custom_title = "Test de l'Espace UMAP"

        # WHEN :
        fig = LatentSpaceVisualizer.plot_umap_clusters(
            embeddings_np=embeddings_np,
            cluster_labels=cluster_labels,
            title=custom_title
        )

        # THEN
        assert isinstance(fig, go.Figure)
        assert custom_title in fig.layout.title.text
        assert len(fig.data) > 0
        assert fig.layout.updatemenus == () or fig.layout.updatemenus is None

    def test__plot_umap_clusters__generates_interactive_dropdown_with_metadata(self):
        # GIVEN :
        np.random.seed(42)
        embeddings_np = np.random.rand(4, 16)
        cluster_labels = np.array([0, 1, 0, 1])

        mock_metadata = [
            {"trial_metadata": {"name": "Trial_1", "client": {"name": "Agroscope"}}},
            {"trial_metadata": {"name": "Trial_2", "client": {"name": "INRAE"}}},
            {"trial_metadata": {"name": "Trial_1", "client": {"name": "Agroscope"}}},
            {"trial_metadata": {"name": "Trial_3", "client": {"name": "INRAE"}}}
        ]

        # WHEN :
        fig = LatentSpaceVisualizer.plot_umap_clusters(
            embeddings_np=embeddings_np,
            cluster_labels=cluster_labels,
            metadata=mock_metadata
        )

        # THEN
        assert isinstance(fig, go.Figure)

        assert fig.layout.updatemenus is not None
        assert len(fig.layout.updatemenus) == 1

        buttons = fig.layout.updatemenus[0].buttons
        button_labels = [b.label for b in buttons]

        assert "Colorer par : Cluster" in button_labels
        assert "Colorer par : Trial Name" in button_labels
        assert "Colorer par : Client" in button_labels


class TestTrainingVisualizer:

    def test__save_loss_curve__creates_html_file_from_valid_json(
            self, tmp_path
    ):
        # GIVEN
        output_dir = tmp_path / "run_artifacts"
        json_path = tmp_path / "metrics.json"

        metrics_payload = {
            "train_loss": [4.34, 4.30, 4.34],
            "val_loss": [4.75, 4.73, 4.70],
        }

        with open(json_path, "w") as f:
            json.dump(metrics_payload, f)

        # WHEN
        TrainingVisualizer.save_loss_curve(str(json_path), str(output_dir))

        # THEN
        expected_html_file = output_dir / "loss_curve.html"
        assert expected_html_file.exists()
        assert expected_html_file.stat().st_size > 0

    def test__save_loss_curve__raises_file_not_found_on_missing_json(
            self, tmp_path
    ):
        # GIVEN
        missing_json_path = tmp_path / "non_existent_file.json"
        output_dir = tmp_path / "output"

        # WHEN / THEN
        with pytest.raises(FileNotFoundError):
            TrainingVisualizer.save_loss_curve(
                str(missing_json_path), str(output_dir)
            )


class TestAnomalyVisualizer:

    def test__plot_signal_and_anomaly__returns_valid_figure_and_saves_html(self, tmp_path):
        # GIVEN
        dates = pd.date_range("2026-01-01", periods=10, freq="5min")

        df_signal = pd.DataFrame({"mV": np.random.randn(10)}, index=dates)

        df_ml = pd.DataFrame({
            "timestamp": dates,
            "score": np.random.uniform(0, 1, 10)
        })

        plant_id = "test_plant_1"
        output_file = tmp_path / "anomaly_plot.html"

        # WHEN
        fig = AnomalyVisualizer.plot_signal_and_anomaly(
            df_signal=df_signal,
            df_ml=df_ml,
            plant_id=plant_id,
            output_html_path=str(output_file)
        )

        # THEN
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2

        assert fig.data[0].name == f"Signal {plant_id} (mV)"
        assert "Distance" in fig.data[1].name

        assert getattr(fig.data[1], "yaxis", None) == "y2"

        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test__plot_signal_with_heatmap__returns_valid_figure_and_subplots(self, tmp_path):
        # GIVEN
        dates = pd.date_range("2026-01-01", periods=10, freq="5min")

        df_signal = pd.DataFrame({"mV": np.random.randn(10)}, index=dates)

        df_ml = pd.DataFrame({
            "timestamp": dates,
            "score": np.random.uniform(0, 1, 10)
        })

        plant_id = "test_plant_2"
        output_file = tmp_path / "heatmap_plot.html"

        # WHEN
        fig = AnomalyVisualizer.plot_signal_with_heatmap(
            df_signal=df_signal,
            df_ml=df_ml,
            plant_id=plant_id,
            output_html_path=str(output_file)
        )

        # THEN
        assert isinstance(fig, go.Figure)

        # Vérification des sous-graphiques (2 traces)
        assert len(fig.data) == 2
        assert fig.data[0].type == "scatter"
        assert fig.data[1].type == "heatmap"

        assert fig.data[0].name == f"Signal {plant_id} (mV)"

        assert output_file.exists()
        assert output_file.stat().st_size > 0


class TestDownstreamTaskVisualizer:

    def test__plot_stress_probability_dynamics__returns_valid_figure_with_vrect(self):
        # GIVEN
        dates = pd.date_range("2026-01-01", periods=10, freq="6h")
        df_trial = pd.DataFrame({
            "prob_control": np.random.uniform(0, 0.2, 10),
            "prob_stress": np.random.uniform(0.8, 1.0, 10)
        }, index=dates)

        trial_name = "Test_Drought"
        stress_start = pd.Timestamp("2026-01-02 00:00:00")

        # WHEN
        fig = DownstreamTaskVisualizer.plot_stress_probability_dynamics(
            df_trial=df_trial,
            trial_name=trial_name,
            stress_start_time=stress_start
        )

        # THEN
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # Deux courbes (Control et Stress)
        assert fig.data[0].name == 'Groupe Contrôle'
        assert fig.data[1].name == 'Groupe Stressé'

        assert fig.layout.shapes is not None
        assert len(fig.layout.shapes) == 1
        assert fig.layout.shapes[0].type == "rect"

    def test__plot_confusion_matrix__returns_valid_heatmap(self):
        # GIVEN
        cm = np.array([[50, 5], [10, 35]])
        classes = ["Healthy", "Stressed"]
        title = "Matrice de Test"

        # WHEN
        fig = DownstreamTaskVisualizer.plot_confusion_matrix(
            conf_matrix=cm,
            class_names=classes,
            title=title
        )

        # THEN
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == title
        assert fig.data[0].type == "heatmap"
        # Vérifie que les axes sont bien nommés
        assert fig.layout.xaxis.title.text == "Classe Prédite"
        assert fig.layout.yaxis.title.text == "Classe Réelle"

    def test__plot_decision_thresholds__returns_valid_figure(self):
        # GIVEN
        curves_data = {
            "thresholds": [0.2, 0.5, 0.8],
            "f1_scores": [0.6, 0.9, 0.7],
            "false_negatives": [2, 5, 20],
            "false_positives": [30, 5, 1],
            "total_positives": 100
        }

        # WHEN
        fig = DownstreamTaskVisualizer.plot_decision_thresholds(curves_data)

        # THEN
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 3  # F1, FN, FP
        assert fig.data[0].name == 'F1-Score'

    def test__plot_roc_curve__returns_valid_figure(self):
        # GIVEN
        curves_data = {"fpr": [0.0, 0.1, 1.0], "tpr": [0.0, 0.9, 1.0]}
        roc_auc = 0.95

        # WHEN
        fig = DownstreamTaskVisualizer.plot_roc_curve(curves_data, roc_auc)

        # THEN
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2
        assert "AUC = 0.9500" in fig.data[0].name

    def test__plot_optimization_heatmap__returns_valid_figure(self):
        # GIVEN
        grid_data = {
            "f1_matrix": [[0.5, 0.6], [0.7, 0.8]],
            "tp_matrix": [[5, 6], [7, 8]],
            "tn_matrix": [[50, 60], [70, 80]],
            "fp_matrix": [[1, 2], [3, 4]],
            "fn_matrix": [[2, 1], [4, 3]],
            "thresholds": [0.3, 0.7],
            "percentiles": [75, 90],
            "best_params": {"percentile": 90, "threshold": 0.7, "f1_score": 0.8, "tp": 8, "fp": 4}
        }

        # WHEN
        fig = DownstreamTaskVisualizer.plot_optimization_heatmap(grid_data)

        # THEN
        assert isinstance(fig, go.Figure)
        assert fig.data[0].type == "heatmap"
        assert "Optimization Grid: F1-Score" in fig.layout.title.text