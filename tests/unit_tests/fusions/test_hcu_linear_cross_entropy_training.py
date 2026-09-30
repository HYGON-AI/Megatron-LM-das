# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Check scheduler loss and gradients against ordinary PyTorch CE on CPU."""

import unittest
from unittest import mock

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required for numerical tests")
class TestTrainingLoss(unittest.TestCase):
    def test_loss_gradients_and_accumulation(self):
        from hcu_megatron.core.fusions import fused_linear_cross_entropy as api

        def native(hidden, weight, labels, **kwargs):
            return torch.nn.functional.cross_entropy(
                (hidden @ weight.T).reshape(-1, weight.shape[0]),
                labels.reshape(-1),
                ignore_index=kwargs["ignore_index"],
                reduction="mean",
            )

        for all_masked in (False, True):
            with self.subTest(all_masked=all_masked):
                torch.manual_seed(42)
                hidden = torch.randn(3, 2, 4, requires_grad=True)
                weight = torch.randn(7, 4, requires_grad=True)
                ref_hidden = hidden.detach().clone().requires_grad_()
                ref_weight = weight.detach().clone().requires_grad_()
                labels = torch.tensor([[0, 1, -100], [3, 4, 5]])
                mask = torch.zeros(2, 3) if all_masked else torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
                with mock.patch.object(api, "linear_cross_entropy", side_effect=native):
                    for _ in range(2):
                        losses = api.linear_cross_entropy_for_training(hidden, weight, labels, mask)
                        actual = (losses * mask).sum()
                        reference = (
                            torch.nn.functional.cross_entropy(
                                (ref_hidden @ ref_weight.T).reshape(-1, 7),
                                labels.T.contiguous().reshape(-1),
                                reduction="none",
                                ignore_index=-100,
                            )
                            .reshape(3, 2)
                            .T
                        )
                        expected = (reference * mask).sum()
                        torch.testing.assert_close(actual, expected)
                        actual.backward()
                        expected.backward()
                        torch.testing.assert_close(hidden.grad, ref_hidden.grad)
                        torch.testing.assert_close(weight.grad, ref_weight.grad)

    def test_empty_batch_rejected(self):
        from hcu_megatron.core.fusions import fused_linear_cross_entropy as api

        with self.assertRaisesRegex(ValueError, "at least one token"):
            api.linear_cross_entropy_for_training(
                torch.empty(0, 1, 3),
                torch.ones(4, 3),
                torch.empty(1, 0, dtype=torch.long),
                torch.empty(1, 0),
            )

    def test_weighted_mask_rejected(self):
        from hcu_megatron.core.fusions import fused_linear_cross_entropy as api

        with self.assertRaisesRegex(RuntimeError, "binary loss_mask"):
            api.linear_cross_entropy_for_training(
                torch.ones(2, 1, 3),
                torch.ones(4, 3),
                torch.zeros(1, 2, dtype=torch.long),
                torch.tensor([[1.0, 0.5]]),
            )


if __name__ == "__main__":
    unittest.main()
