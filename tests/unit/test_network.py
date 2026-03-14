"""
tests/unit/test_network.py — Unit tests for src/network.py.

Verifies:
  - QNetwork forward pass produces output tensors of the correct shape.
  - Output dtype is float32.
  - Network parameters have the expected layer structure.
  - Gradient flow is intact (backward pass does not error).
"""

import pytest
import torch

from config import HIDDEN_SIZE, N_ACTIONS, OBS_DIM
from network import QNetwork


class TestQNetworkConstruction:
    """Tests for default and custom constructor arguments."""

    def test_default_construction(self):
        net = QNetwork()
        assert net is not None

    def test_custom_dimensions(self):
        net = QNetwork(obs_dim=5, n_actions=2, hidden_size=64)
        out = net(torch.zeros(1, 5))
        assert out.shape == (1, 2)

    def test_parameter_count(self):
        """Verify the network has the expected number of trainable parameters.

        Layer shapes:
            Linear(11, 256): 11*256 + 256 = 3072
            Linear(256, 256): 256*256 + 256 = 65792
            Linear(256, 3):  256*3 + 3 = 771
            Total: 69635
        """
        net = QNetwork()
        total = sum(p.numel() for p in net.parameters() if p.requires_grad)
        expected = (
            (OBS_DIM * HIDDEN_SIZE + HIDDEN_SIZE)
            + (HIDDEN_SIZE * HIDDEN_SIZE + HIDDEN_SIZE)
            + (HIDDEN_SIZE * N_ACTIONS + N_ACTIONS)
        )
        assert total == expected


class TestQNetworkForward:
    """Tests for the forward pass output shape and dtype."""

    @pytest.fixture(autouse=True)
    def _net(self):
        self.net = QNetwork()

    def test_output_shape_batch_1(self):
        x = torch.zeros(1, OBS_DIM)
        out = self.net(x)
        assert out.shape == (1, N_ACTIONS)

    def test_output_shape_batch_8(self):
        x = torch.zeros(8, OBS_DIM)
        out = self.net(x)
        assert out.shape == (8, N_ACTIONS)

    def test_output_shape_batch_64(self):
        x = torch.zeros(64, OBS_DIM)
        out = self.net(x)
        assert out.shape == (64, N_ACTIONS)

    def test_output_dtype_float32(self):
        x = torch.zeros(4, OBS_DIM, dtype=torch.float32)
        out = self.net(x)
        assert out.dtype == torch.float32

    def test_output_no_nan(self):
        torch.manual_seed(0)
        x = torch.randn(16, OBS_DIM)
        out = self.net(x)
        assert not torch.isnan(out).any()

    def test_output_not_all_zeros_for_random_input(self):
        """Non-zero input should produce non-zero output (weights are randomly initialised)."""
        torch.manual_seed(42)
        x = torch.randn(4, OBS_DIM)
        out = self.net(x)
        assert out.abs().sum().item() > 0.0


class TestQNetworkGradients:
    """Tests for gradient flow through the network."""

    def test_backward_pass_runs(self):
        net = QNetwork()
        x = torch.randn(4, OBS_DIM)
        out = net(x)
        loss = out.sum()
        loss.backward()  # Must not raise.

    def test_gradients_exist_after_backward(self):
        net = QNetwork()
        x = torch.randn(4, OBS_DIM)
        out = net(x)
        out.sum().backward()
        for name, param in net.named_parameters():
            assert param.grad is not None, f"No gradient for parameter: {name}"

    def test_gradient_non_zero_after_backward(self):
        """At least one parameter gradient must be non-zero after a real loss."""
        torch.manual_seed(7)
        net = QNetwork()
        x = torch.randn(4, OBS_DIM)
        target = torch.randn(4, N_ACTIONS)
        import torch.nn.functional as F

        loss = F.mse_loss(net(x), target)
        loss.backward()
        grad_norms = [p.grad.norm().item() for p in net.parameters() if p.grad is not None]
        assert any(g > 0.0 for g in grad_norms)
