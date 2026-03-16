"""
tests/unit/test_utils.py — Unit tests for src/utils.py.

Verifies:
  - set_seed rejects negative seeds.
  - set_seed produces reproducible Python random output.
  - set_seed produces reproducible NumPy output.
  - set_seed produces reproducible PyTorch CPU output.
  - set_seed produces identical PyTorch output across two calls with the same seed.
  - set_seed runs without error on whatever device is available (CUDA or CPU).
"""

import random

import numpy as np
import pytest
import torch

from utils import set_seed


class TestSetSeedValidation:

    def test_negative_seed_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            set_seed(-1)

    def test_zero_seed_is_valid(self):
        set_seed(0)  # Must not raise.

    def test_large_seed_is_valid(self):
        set_seed(2**31 - 1)  # Must not raise.


class TestSetSeedReproducibility:

    def test_python_random_reproducible(self):
        set_seed(7)
        a = [random.random() for _ in range(5)]
        set_seed(7)
        b = [random.random() for _ in range(5)]
        assert a == b

    def test_numpy_random_reproducible(self):
        set_seed(7)
        a = np.random.standard_normal(5)
        set_seed(7)
        b = np.random.standard_normal(5)
        np.testing.assert_array_equal(a, b)

    def test_torch_cpu_reproducible(self):
        set_seed(7)
        a = torch.randn(5)
        set_seed(7)
        b = torch.randn(5)
        assert torch.equal(a, b)

    def test_different_seeds_produce_different_torch_output(self):
        set_seed(7)
        a = torch.randn(10)
        set_seed(99)
        b = torch.randn(10)
        assert not torch.equal(a, b)


class TestSetSeedCUDA:

    def test_runs_without_error_regardless_of_cuda(self):
        """set_seed must complete successfully on both CUDA and CPU machines."""
        set_seed(42)  # Must not raise whether CUDA is present or absent.

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available on this machine")
    def test_cuda_rng_reproducible(self):
        """With CUDA, two calls with the same seed produce identical GPU tensors."""
        set_seed(42)
        a = torch.randn(10, device="cuda")
        set_seed(42)
        b = torch.randn(10, device="cuda")
        assert torch.equal(a, b)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available on this machine")
    def test_different_seeds_produce_different_cuda_output(self):
        set_seed(42)
        a = torch.randn(10, device="cuda")
        set_seed(99)
        b = torch.randn(10, device="cuda")
        assert not torch.equal(a, b)
