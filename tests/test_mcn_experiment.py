import unittest

import torch

from mcn.experiment import graph_to_dot
from mcn.model import MCNConfig, MCNSeq2Seq


class MCNExperimentTests(unittest.TestCase):
    def test_graph_to_dot_exports_active_cells_and_edges(self):
        torch.manual_seed(0)
        config = MCNConfig(d_model=32, n_cells=8, n_seed_cells=4, n_dev_steps=2, d_signal=16, max_seq_len=8)
        model = MCNSeq2Seq(src_vocab_size=10, tgt_vocab_size=9, config=config)
        src = torch.tensor([[1, 2, 3, 0]])

        model.eval()
        with torch.no_grad():
            _, _, _, core_out = model.encode(src)
        dot = graph_to_dot(core_out, [op.__class__.__name__ for op in model.core.ops], edge_threshold=0.0)

        self.assertIn("digraph MCN", dot)
        self.assertIn("cell 0", dot)
        self.assertIn("->", dot)


if __name__ == "__main__":
    unittest.main()
