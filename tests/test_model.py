import unittest

try:
    import torch

    from activity_patterns.labels import CATEGORY_NAMES, CATEGORY_TO_NUM_FINE
    from activity_patterns.model import (
        HierarchicalProtocolEventLSTM,
        ProtocolEventLSTM,
        hierarchical_loss,
    )
except ImportError:  # pragma: no cover - optional model dependency
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class ModelTests(unittest.TestCase):
    def test_flat_lstm_forward_shape(self):
        model = ProtocolEventLSTM(
            vocab_size=50,
            num_classes=8,
            embedding_dim=16,
            hidden_dim=32,
            num_layers=1,
        )
        token_ids = torch.randint(1, 50, (2, 5, 4))
        lengths = torch.tensor([5, 3])

        logits = model(token_ids, lengths)

        self.assertEqual(tuple(logits.shape), (2, 8))

    def test_hierarchical_lstm_forward_and_loss(self):
        model = HierarchicalProtocolEventLSTM(
            vocab_size=50,
            category_to_num_fine=CATEGORY_TO_NUM_FINE,
            embedding_dim=16,
            hidden_dim=32,
            num_layers=1,
        )
        token_ids = torch.randint(1, 50, (3, 5, 4))
        lengths = torch.tensor([5, 4, 3])

        outputs = model(token_ids, lengths)

        self.assertEqual(tuple(outputs["coarse"].shape), (3, len(CATEGORY_NAMES)))
        self.assertEqual(
            tuple(outputs["fine"]["Recon"].shape),
            (3, CATEGORY_TO_NUM_FINE["Recon"]),
        )

        coarse_targets = torch.tensor([0, 1, 3])
        fine_targets = torch.tensor([0, 2, 4])
        loss = hierarchical_loss(
            outputs,
            coarse_targets,
            fine_targets,
            CATEGORY_NAMES,
        )

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
