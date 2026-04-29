import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mcn.config_io import flatten_cli_args, load_config, save_config


class MCNConfigIOTests(unittest.TestCase):
    def test_save_load_and_flatten_config(self):
        payload = {
            "model": {"model": "mcn", "d_model": 32, "hard_ops": True},
            "training": {"epochs": 1},
            "data": {"split": "random"},
        }
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            save_config(path, payload)
            loaded = load_config(path)

        args = flatten_cli_args(loaded)
        self.assertEqual(loaded["model"]["d_model"], 32)
        self.assertIn("--d-model", args)
        self.assertIn("32", args)
        self.assertIn("--hard-ops", args)


if __name__ == "__main__":
    unittest.main()
