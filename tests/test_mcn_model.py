import unittest

import torch
import torch.nn.functional as F

from mcn.model import (
    FixedClassifierBaseline,
    FixedSeq2SeqBaseline,
    MCNConfig,
    MCNClassifier,
    MCNSeq2Seq,
    TransformerClassifierBaseline,
    TransformerSeq2SeqBaseline,
    mcn_regularization,
)


class MCNModelTests(unittest.TestCase):
    def test_classifier_forward_and_backward(self):
        torch.manual_seed(0)
        config = MCNConfig(d_model=32, n_cells=8, n_dev_steps=3, d_signal=16, max_seq_len=8)
        model = MCNClassifier(vocab_size=12, n_classes=4, config=config)
        tokens = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
        labels = torch.tensor([1, 3])

        logits, core_out = model(tokens)
        loss = F.cross_entropy(logits, labels) + mcn_regularization(core_out)
        loss.backward()

        self.assertEqual(logits.shape, (2, 4))
        self.assertEqual(core_out.cell_states.shape, (2, 8, 32))
        self.assertIn("development_steps", core_out.metrics)
        self.assertGreater(model.token_emb.weight.grad.abs().sum().item(), 0)

    def test_seq2seq_with_cgb_forward(self):
        torch.manual_seed(0)
        config = MCNConfig(
            d_model=32,
            n_cells=8,
            n_seed_cells=4,
            n_dev_steps=2,
            d_signal=16,
            use_cgb=True,
            clifford_p=3,
            max_seq_len=8,
        )
        model = MCNSeq2Seq(src_vocab_size=10, tgt_vocab_size=9, config=config)
        src = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
        tgt_in = torch.tensor([[1, 3, 4], [1, 5, 0]])

        logits, core_out = model(src, tgt_in)

        self.assertEqual(logits.shape, (2, 3, 9))
        self.assertEqual(core_out.adjacency.shape, (2, 8, 8))
        self.assertIn("spawned_cells", core_out.metrics)

    def test_proliferation_activates_dormant_slots(self):
        torch.manual_seed(0)
        config = MCNConfig(d_model=32, n_cells=12, n_seed_cells=3, n_dev_steps=3, d_signal=16, max_seq_len=8)
        model = MCNClassifier(vocab_size=12, n_classes=4, config=config)
        tokens = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])

        _, core_out = model(tokens)

        self.assertGreater(core_out.metrics["occupied_cells"].item(), config.n_seed_cells)

    def test_proliferation_can_be_disabled(self):
        torch.manual_seed(0)
        config = MCNConfig(
            d_model=32,
            n_cells=12,
            n_seed_cells=3,
            n_dev_steps=3,
            d_signal=16,
            enable_proliferation=False,
            max_seq_len=8,
        )
        model = MCNClassifier(vocab_size=12, n_classes=4, config=config)
        tokens = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])

        _, core_out = model(tokens)

        self.assertAlmostEqual(core_out.metrics["occupied_cells"].item(), float(config.n_seed_cells), places=5)

    def test_eval_forward_is_deterministic(self):
        torch.manual_seed(0)
        config = MCNConfig(d_model=32, n_cells=8, n_dev_steps=2, d_signal=16, dropout=0.0, max_seq_len=8)
        model = MCNClassifier(vocab_size=12, n_classes=4, config=config)
        model.eval()
        tokens = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])

        with torch.no_grad():
            first, _ = model(tokens)
            second, _ = model(tokens)

        self.assertTrue(torch.allclose(first, second))

    def test_sequence_length_guard(self):
        config = MCNConfig(d_model=32, n_cells=8, n_dev_steps=2, d_signal=16, max_seq_len=3)
        model = MCNClassifier(vocab_size=12, n_classes=4, config=config)

        with self.assertRaisesRegex(ValueError, "exceeds max_seq_len"):
            model(torch.tensor([[1, 2, 3, 4]]))

    def test_fixed_seq2seq_baseline_forward_and_generate(self):
        torch.manual_seed(0)
        config = MCNConfig(d_model=32, n_cells=8, n_dev_steps=2, d_signal=16, max_seq_len=8)
        model = FixedSeq2SeqBaseline(src_vocab_size=10, tgt_vocab_size=9, config=config)
        src = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
        tgt_in = torch.tensor([[1, 3, 4], [1, 5, 0]])

        logits = model(src, tgt_in)
        pred = model.generate(src, bos_id=1, eos_id=2, max_len=5)

        self.assertEqual(logits.shape, (2, 3, 9))
        self.assertEqual(pred.shape[0], 2)

    def test_transformer_seq2seq_baseline_forward_and_generate(self):
        torch.manual_seed(0)
        config = MCNConfig(d_model=32, n_cells=8, n_dev_steps=2, d_signal=16, max_seq_len=8)
        model = TransformerSeq2SeqBaseline(src_vocab_size=10, tgt_vocab_size=9, config=config)
        src = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
        tgt_in = torch.tensor([[1, 3, 4], [1, 5, 0]])

        logits = model(src, tgt_in)
        pred = model.generate(src, bos_id=1, eos_id=2, max_len=5)

        self.assertEqual(logits.shape, (2, 3, 9))
        self.assertEqual(pred.shape[0], 2)

    def test_fixed_classifier_baseline_forward(self):
        torch.manual_seed(0)
        config = MCNConfig(d_model=32, n_cells=8, n_dev_steps=2, d_signal=16, max_seq_len=8)
        model = FixedClassifierBaseline(vocab_size=12, n_classes=4, config=config)
        tokens = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])

        logits = model(tokens)

        self.assertEqual(logits.shape, (2, 4))

    def test_transformer_classifier_baseline_forward(self):
        torch.manual_seed(0)
        config = MCNConfig(d_model=32, n_cells=8, n_dev_steps=2, d_signal=16, max_seq_len=8)
        model = TransformerClassifierBaseline(vocab_size=12, n_classes=4, config=config)
        tokens = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])

        logits = model(tokens)

        self.assertEqual(logits.shape, (2, 4))

    def test_different_inputs_produce_different_graphs(self):
        torch.manual_seed(0)
        config = MCNConfig(
            d_model=32,
            n_cells=10,
            n_seed_cells=4,
            n_dev_steps=3,
            n_exec_steps=2,
            d_signal=16,
            dropout=0.0,
            max_seq_len=8,
        )
        model = MCNClassifier(vocab_size=20, n_classes=4, config=config)
        model.eval()
        first = torch.tensor([[1, 2, 3, 0]])
        second = torch.tensor([[7, 8, 9, 10]])

        with torch.no_grad():
            _, core_a = model(first)
            _, core_b = model(second)

        self.assertFalse(torch.allclose(core_a.adjacency, core_b.adjacency))
        self.assertFalse(torch.allclose(core_a.active_gates, core_b.active_gates))

    def test_development_trace_is_recorded_when_enabled(self):
        torch.manual_seed(0)
        config = MCNConfig(
            d_model=32,
            n_cells=8,
            n_dev_steps=3,
            d_signal=16,
            save_development_trace=True,
            max_seq_len=8,
        )
        model = MCNClassifier(vocab_size=12, n_classes=4, config=config)
        tokens = torch.tensor([[1, 2, 3, 0]])

        _, core_out = model(tokens)

        self.assertIsNotNone(core_out.development_trace)
        self.assertEqual(len(core_out.development_trace["adjacency"]), config.n_dev_steps)
        self.assertIn("adjacency_entropy", core_out.metrics)

    def test_regularization_increases_with_dense_adjacency(self):
        torch.manual_seed(0)
        config = MCNConfig(d_model=32, n_cells=8, n_dev_steps=2, d_signal=16, max_seq_len=8)
        model = MCNClassifier(vocab_size=12, n_classes=4, config=config)
        tokens = torch.tensor([[1, 2, 3, 0]])
        _, core_out = model(tokens)

        sparse_loss = mcn_regularization(core_out, compute_weight=0.0, stability_weight=0.0, diversity_weight=0.0, sparsity_weight=1.0)
        dense_core = core_out
        dense_core.adjacency = torch.ones_like(core_out.adjacency)
        dense_loss = mcn_regularization(dense_core, compute_weight=0.0, stability_weight=0.0, diversity_weight=0.0, sparsity_weight=1.0)

        self.assertGreater(dense_loss.item(), sparse_loss.item())

    def test_curriculum_freedom_ramp(self):
        config = MCNConfig(d_model=32, n_cells=8, n_dev_steps=2, d_signal=16, max_seq_len=8)
        core = MCNClassifier(vocab_size=12, n_classes=4, config=config).core

        # Default: full freedom
        self.assertAlmostEqual(core.curriculum_freedom, 1.0)

        # progress=0 -> freedom=0 (first 10%)
        core.set_curriculum_progress(0.0)
        self.assertAlmostEqual(core.curriculum_freedom, 0.0)
        core.set_curriculum_progress(0.05)
        self.assertAlmostEqual(core.curriculum_freedom, 0.0)

        # progress=0.2 -> freedom=0.5 (linear ramp between 10%-30%)
        core.set_curriculum_progress(0.2)
        self.assertAlmostEqual(core.curriculum_freedom, 0.5)

        # progress=0.3+ -> freedom=1.0
        core.set_curriculum_progress(0.3)
        self.assertAlmostEqual(core.curriculum_freedom, 1.0)
        core.set_curriculum_progress(1.0)
        self.assertAlmostEqual(core.curriculum_freedom, 1.0)

    def test_curriculum_constrains_early_development(self):
        torch.manual_seed(0)
        config = MCNConfig(d_model=32, n_cells=12, n_seed_cells=3, n_dev_steps=3, d_signal=16, max_seq_len=8)
        model = MCNClassifier(vocab_size=12, n_classes=4, config=config)
        tokens = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])

        # Full freedom run
        model.core.set_curriculum_progress(1.0)
        _, core_free = model(tokens)
        free_spawned = core_free.metrics["spawned_cells"].item()

        # Constrained run (progress=0 -> freedom=0)
        model.core.set_curriculum_progress(0.0)
        _, core_constrained = model(tokens)
        constrained_spawned = core_constrained.metrics["spawned_cells"].item()

        # Curriculum should reduce spawning at progress=0
        self.assertLessEqual(constrained_spawned, free_spawned + 0.01)

    def test_cgb_higher_grade_binding_contributes(self):
        torch.manual_seed(0)
        from mcn.model import CGBInputAttentionOp
        op = CGBInputAttentionOp(d_model=32, dropout=0.0, clifford_p=3)
        op.eval()
        states = torch.randn(2, 4, 32)
        memory = torch.randn(2, 6, 32)

        output = op(states, memory)
        self.assertEqual(output.shape, (2, 4, 32))

        # Verify binding_proj exists and its parameters get gradients
        op.train()
        loss = op(states, memory).sum()
        loss.backward()
        self.assertGreater(op.binding_proj.weight.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
