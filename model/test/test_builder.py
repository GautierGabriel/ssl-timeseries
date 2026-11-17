import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from vivml.SSL.model.builder import build_model_from_config, get_class_from_module
from vivml.SSL.model.builder import init_model_weights


class TestModelBuilder:

    def test__get_class_from_module__raises_error_for_unknown_class(self):
        with pytest.raises(ValueError, match="Class 'NoEncoder' not found"):
            get_class_from_module("NoEncoder")

    @patch('vivml.SSL.model.builder.get_class_from_module')
    def test__build_model_from_config__instantiates_method_and_components(self, mock_get_class):
        config = {
            "data": {
                "input_features": {
                    "Micro_Stress": {
                        "normalizers": {
                            "type": "MockNormalizer",
                            "epsilon": 1e-6
                        }
                    },
                    "raw": None
                }
            },
            "model": {
                "method": {
                    "type": "MockMethod",
                    "freeze_encoder": True
                },
                "encoder": {
                    "type": "MockEncoder",
                    "in_channels": 1,
                    "output_dim": 128
                },
                "head": {
                    "type": "MockHead",
                    "num_classes": 2
                }
            }
        }

        MockMethodClass = MagicMock()
        MockEncoderClass = MagicMock()

        class SpyHead:
            called_with = None

            def __init__(self, input_dim, num_classes):
                SpyHead.called_with = {"input_dim": input_dim, "num_classes": num_classes}

        class SpyNormalizer(nn.Module):
            called_with = None

            def __init__(self, epsilon):
                super().__init__()
                SpyNormalizer.called_with = {"epsilon": epsilon}

        mock_encoder_instance = MagicMock()
        mock_encoder_instance.output_dim = 128
        MockEncoderClass.return_value = mock_encoder_instance

        def side_effect(class_name):
            if class_name == "MockMethod": return MockMethodClass
            if class_name == "MockEncoder": return MockEncoderClass
            if class_name == "MockHead": return SpyHead
            if class_name == "MockNormalizer": return SpyNormalizer
            return None

        mock_get_class.side_effect = side_effect

        _ = build_model_from_config(config)

        MockEncoderClass.assert_called_once_with(in_channels=1, output_dim=128)
        assert SpyHead.called_with == {"input_dim": 128, "num_classes": 2}
        assert SpyNormalizer.called_with == {"epsilon": 1e-6}

        args, kwargs = MockMethodClass.call_args
        assert kwargs["freeze_encoder"] is True
        assert kwargs["encoder"] == mock_encoder_instance
        assert isinstance(kwargs["head"], SpyHead)

        assert kwargs["target_channels"] == ["Micro_Stress", "raw"]
        assert isinstance(kwargs["normalizers"], nn.ModuleDict)
        assert "Micro_Stress" in kwargs["normalizers"]
        assert isinstance(kwargs["normalizers"]["Micro_Stress"], SpyNormalizer)

    @patch('os.path.exists')
    @patch('torch.load')
    def test__init_model_weights__loads_pretrained_weights_if_present(self, mock_torch_load, mock_path_exists):

        mock_path_exists.return_value = True

        config = {
            "model": {
                "method": {
                    "pretrained_weights": "/fake/path/weights.pt"
                }
            }
        }

        mock_model = MagicMock()
        mock_encoder_instance = MagicMock()
        mock_model.encoder = mock_encoder_instance

        fake_state_dict = {"encoder.conv1.weight": torch.tensor([1.0])}
        mock_torch_load.return_value = fake_state_dict

        # ACTION
        init_model_weights(mock_model, config)

        # ASSERTIONS
        mock_torch_load.assert_called_once_with("/fake/path/weights.pt", map_location="cpu")

        expected_encoder_weights = {"conv1.weight": fake_state_dict["encoder.conv1.weight"]}
        mock_model.encoder.load_state_dict.assert_called_once_with(expected_encoder_weights, strict=False)