import unittest

from mcn.toy_scan import build_vocabs, collate_examples, make_jump_composition_split, make_random_split


class ToyScanTests(unittest.TestCase):
    def test_jump_split_and_collate(self):
        train, test = make_jump_composition_split()
        src_vocab, tgt_vocab = build_vocabs([*train, *test])
        batch = collate_examples(train[:4], src_vocab, tgt_vocab)

        self.assertTrue(train)
        self.assertTrue(test)
        self.assertTrue(all(example.command == "jump" or "jump" not in example.command.split() for example in train))
        self.assertTrue(any("jump" in example.command.split() and example.command != "jump" for example in test))
        self.assertIn("src", batch)
        self.assertIn("tgt_in", batch)
        self.assertIn("tgt_out", batch)

    def test_random_split_is_seeded_and_nonempty(self):
        first_train, first_test = make_random_split(seed=3, test_fraction=0.1)
        second_train, second_test = make_random_split(seed=3, test_fraction=0.1)

        self.assertTrue(first_train)
        self.assertTrue(first_test)
        self.assertEqual(first_train, second_train)
        self.assertEqual(first_test, second_test)


if __name__ == "__main__":
    unittest.main()
