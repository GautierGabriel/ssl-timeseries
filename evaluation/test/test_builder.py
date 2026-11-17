import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from vivml.SSL.evaluation.builder import run_full_evaluation_pipeline


class TestEvaluationBuilder:

    @patch('vivml.SSL.evaluation.builder.build_evaluator_from_config')
    @patch('json.dump')
    @patch('builtins.open')
    def test__run_full_evaluation_pipeline__orchestrates_supervised_flow(self, mock_open, mock_json, mock_build):
        config = {"evaluation": {"type": "SupervisedInference"}}
        mock_model = MagicMock()
        mock_loader = MagicMock()

        # 1. SETUP du Mock Engine
        mock_engine = MagicMock()

        raw_results = {
            "outputs": np.array([[0.8, 0.2], [0.1, 0.9]]),
            "labels": np.array([0, 1])
        }
        mock_engine.run_inference.return_value = raw_results

        # Le mock renvoie exactement la structure définie dans ta classe SupervisedInference
        mock_engine.compute_metrics.return_value = {
            "scalars": {
                "accuracy": 1.0,
                "f1_score": 1.0
            },
            "curves": {},
            "metadata": []
        }

        mock_build.return_value = mock_engine

        # 2. EXECUTION
        final_metrics = run_full_evaluation_pipeline(
            config=config, model=mock_model, val_loader=mock_loader, device="cpu", run_dir="/fake/dir"
        )

        # 3. ASSERTIONS
        mock_build.assert_called_once_with(config, "cpu")
        mock_engine.run_inference.assert_called_once_with(mock_model, mock_loader)
        mock_engine.compute_metrics.assert_called_once_with(raw_results)

        # On vérifie la valeur à l'intérieur de la clé 'scalars'
        assert final_metrics["scalars"]["accuracy"] == 1.0