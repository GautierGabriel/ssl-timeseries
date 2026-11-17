import pytest
import torch
from vivml.SSL.model.loss.losses import MSELoss, MAELoss, HuberLoss, NTXentLoss, WeightedCrossEntropy, SimpleCrossEntropy
import torch.nn.functional as F

class TestNTXentLoss:

    def test__forward__returns_scalar_tensor_with_correct_dimensions(self):
        # GIVEN
        batch_size = 8
        dim = 32
        z = torch.rand(2 * batch_size, dim)
        loss_fn = NTXentLoss(temperature=0.1)

        # WHEN
        loss = loss_fn(z)

        # THEN
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

    def test__forward__returns_expected_low_loss_range_on_perfect_alignment(self):
        # GIVEN :
        z = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=torch.float32)

        loss_fn = NTXentLoss(temperature=1.0)

        expected_low_loss = 1.1633

        # WHEN
        loss = loss_fn(z)

        # THEN :
        assert loss.item() == pytest.approx(expected_low_loss, rel=1e-2, abs=1e-2)

    def test__forward__returns_expected_high_loss_range_on_worst_alignment(self):
        # GIVEN :
        z = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=torch.float32)

        loss_fn = NTXentLoss(temperature=1.0)
        expected_high_loss = 2.497

        # WHEN
        loss = loss_fn(z)

        assert loss.item() > 2.0
        assert loss.item() == pytest.approx(expected_high_loss, rel=1e-2, abs=1e-2)


class TestWeightedCrossEntropy:

    def test__forward__returns_scalar_tensor_with_correct_dimensions(self):
        # GIVEN
        batch_size = 4
        num_classes = 2
        z = torch.rand(batch_size, num_classes)
        labels = torch.randint(0, num_classes, (batch_size,))
        loss_fn = WeightedCrossEntropy(weights=[0.5, 0.5])

        # WHEN
        loss = loss_fn(z, labels)

        # THEN
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

    def test__forward__behaves_like_standard_cross_entropy_without_weights(self):
        # GIVEN
        z = torch.tensor([
            [2.0, -1.0],
            [-1.0, 2.0],
        ])
        labels = torch.tensor([0, 1])

        loss_fn_custom = WeightedCrossEntropy(weights=None)

        # WHEN
        custom_loss = loss_fn_custom(z, labels)
        standard_loss = F.cross_entropy(z, labels)

        # THEN
        assert custom_loss.item() == pytest.approx(standard_loss.item())

    def test__forward__applies_weights_correctly_on_worst_prediction(self):
        # GIVEN
        z = torch.tensor([
            [-3.0, 3.0],
        ])
        labels = torch.tensor([0])

        loss_fn_standard = WeightedCrossEntropy(weights=None)
        standard_loss = loss_fn_standard(z, labels).item()

        loss_fn_weighted = WeightedCrossEntropy(weights=[10.0, 1.0])
        weighted_loss = loss_fn_weighted(z, labels).item()

        assert weighted_loss == pytest.approx(standard_loss, rel=1e-4)

    def test__init__converts_list_to_tensor(self):
        # GIVEN
        weights_list = [0.2, 0.8]

        # WHEN
        loss_fn = WeightedCrossEntropy(weights=weights_list)

        # THEN
        assert isinstance(loss_fn.weights, torch.Tensor)
        assert loss_fn.weights.tolist() == pytest.approx(weights_list)


class TestMSELoss:

    def test__forward__returns_correct_scalar_loss(self):
        # GIVEN
        preds = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        targets = torch.tensor([[1.0, 2.0], [3.0, 6.0]])

        loss_fn = MSELoss(reduction='mean')

        # WHEN
        loss = loss_fn(preds, targets)

        # THEN
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() == pytest.approx(1.0)

    def test__forward__raises_value_error_on_shape_mismatch(self):
        # GIVEN
        preds = torch.tensor([[1.0, 2.0]])  # Shape [1, 2]
        targets = torch.tensor([1.0, 2.0])  # Shape [2]

        loss_fn = MSELoss()

        # WHEN / THEN
        with pytest.raises(ValueError, match="Shape mismatch"):
            loss_fn(preds, targets)

    def test__forward__respects_reduction_parameter(self):
        # GIVEN
        preds = torch.tensor([1.0, 2.0])
        targets = torch.tensor([1.0, 4.0])

        loss_fn = MSELoss(reduction='sum')

        # WHEN
        loss = loss_fn(preds, targets)

        # THEN
        assert loss.item() == pytest.approx(4.0)

class TestMAELoss:
    def test__forward__returns_correct_scalar_loss(self):
        # GIVEN
        preds = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        targets = torch.tensor([[1.0, 2.0], [3.0, 6.0]])
        loss_fn = MAELoss(reduction='mean')

        # WHEN
        loss = loss_fn(preds, targets)

        # THEN
        # Erreur absolue: |4.0 - 6.0| = 2.0 / 4 elements = 0.5
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() == pytest.approx(0.5)

    def test__forward__raises_value_error_on_shape_mismatch(self):
        # GIVEN
        preds = torch.tensor([[1.0, 2.0]])
        targets = torch.tensor([1.0, 2.0])
        loss_fn = MAELoss()

        # WHEN / THEN
        with pytest.raises(ValueError, match="Shape mismatch"):
            loss_fn(preds, targets)


class TestHuberLoss:
    def test__forward__returns_correct_scalar_loss(self):
        # GIVEN
        preds = torch.tensor([1.0, 2.0])
        targets = torch.tensor([1.0, 4.0])
        # Delta par défaut = 1.0.
        # Erreur point 1: 0 (mse). Erreur point 2: 2.0 (> delta, donc MAE: 2.0 - 0.5 = 1.5)
        # Moyenne: 1.5 / 2 = 0.75
        loss_fn = HuberLoss(reduction='mean', delta=1.0)

        # WHEN
        loss = loss_fn(preds, targets)

        # THEN
        assert isinstance(loss, torch.Tensor)
        assert loss.item() == pytest.approx(0.75)

    def test__forward__raises_value_error_on_shape_mismatch(self):
        preds = torch.tensor([[1.0, 2.0]])
        targets = torch.tensor([1.0, 2.0])
        loss_fn = HuberLoss()

        with pytest.raises(ValueError, match="Shape mismatch"):
            loss_fn(preds, targets)


class TestSimpleCrossEntropy:

    def test__forward__calculates_standard_cross_entropy_correctly(self):
        loss_fn = SimpleCrossEntropy()

        logits = torch.tensor([
            [2.0, -1.0],
            [-1.0, 2.0],
            [0.0, 0.0],
            [2.0, 2.0]
        ])
        labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)

        expected_loss = F.cross_entropy(logits, labels)

        # WHEN :
        actual_loss = loss_fn(logits, labels)

        # THEN :
        assert isinstance(actual_loss, torch.Tensor)
        assert actual_loss.dim() == 0
        assert torch.allclose(actual_loss, expected_loss)

    def test__backward_pass__propagates_gradients_to_inputs(self):
        # GIVEN :
        loss_fn = SimpleCrossEntropy()
        logits = torch.randn(4, 2, requires_grad=True)
        labels = torch.randint(0, 2, (4,), dtype=torch.long)

        # WHEN :
        loss = loss_fn(logits, labels)
        loss.backward()

        # THEN :
        assert logits.grad is not None
        assert not torch.all(logits.grad == 0.0)