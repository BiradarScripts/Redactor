import unittest

from mcn.synthetic_tasks import (
    TASK_BUILDERS,
    build_classification_vocabs,
    collate_classification,
    make_binding_task,
)


class SyntheticTaskTests(unittest.TestCase):
    def test_all_task_builders_return_train_and_test(self):
        for name, builder in TASK_BUILDERS.items():
            with self.subTest(name=name):
                train, test = builder()
                self.assertTrue(train)
                self.assertTrue(test)

    def test_collate_classification(self):
        train, test = make_binding_task()
        vocab, label_to_id, _ = build_classification_vocabs([*train, *test])
        batch = collate_classification(train[:4], vocab, label_to_id)

        self.assertEqual(batch["tokens"].shape[0], 4)
        self.assertEqual(batch["labels"].shape[0], 4)


if __name__ == "__main__":
    unittest.main()
