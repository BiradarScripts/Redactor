from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

import torch

from .toy_scan import Vocab


@dataclass(frozen=True)
class ClassificationExample:
    tokens: tuple[str, ...]
    label: str


def build_classification_vocabs(
    examples: Iterable[ClassificationExample],
) -> tuple[Vocab, dict[str, int], list[str]]:
    tokens: list[str] = []
    labels: list[str] = []
    for example in examples:
        tokens.extend(example.tokens)
        labels.append(example.label)
    label_names = sorted(set(labels))
    return Vocab(tokens), {label: idx for idx, label in enumerate(label_names)}, label_names


def collate_classification(
    examples: list[ClassificationExample],
    vocab: Vocab,
    label_to_id: dict[str, int],
) -> dict[str, torch.Tensor]:
    encoded = [vocab.encode(example.tokens, add_eos=True) for example in examples]
    max_len = max(len(row) for row in encoded)
    tokens = torch.full((len(examples), max_len), vocab.pad_id, dtype=torch.long)
    labels = torch.empty(len(examples), dtype=torch.long)
    for idx, row in enumerate(encoded):
        tokens[idx, : len(row)] = torch.tensor(row, dtype=torch.long)
        labels[idx] = label_to_id[examples[idx].label]
    return {"tokens": tokens, "labels": labels}


def make_binding_task(seed: int = 7) -> tuple[list[ClassificationExample], list[ClassificationExample]]:
    colors = ["red", "blue", "green", "yellow"]
    shapes = ["circle", "square", "triangle", "star"]
    rng = random.Random(seed)
    train: list[ClassificationExample] = []
    test: list[ClassificationExample] = []

    for left_color in colors:
        for left_shape in shapes:
            for right_color in colors:
                if right_color == left_color:
                    continue
                for right_shape in shapes:
                    tokens = ("obj", left_color, left_shape, "obj", right_color, right_shape, "query", left_color, "shape")
                    train.append(ClassificationExample(tokens, left_shape))

    for query_color in colors:
        others = [color for color in colors if color != query_color]
        for first_shape in shapes:
            for second_shape in shapes:
                for third_shape in shapes:
                    object_specs = [
                        (query_color, first_shape),
                        (others[0], second_shape),
                        (others[1], third_shape),
                    ]
                    rng.shuffle(object_specs)
                    tokens_list = []
                    for color, shape in object_specs:
                        tokens_list.extend(["obj", color, shape])
                    tokens_list.extend(["query", query_color, "shape"])
                    test.append(ClassificationExample(tuple(tokens_list), first_shape))

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def make_length_generalization_task(
    seed: int = 7,
    train_max_len: int = 8,
    test_max_len: int = 16,
) -> tuple[list[ClassificationExample], list[ClassificationExample]]:
    rng = random.Random(seed)
    train: list[ClassificationExample] = []
    test: list[ClassificationExample] = []
    for length in range(3, test_max_len + 1):
        target = train if length <= train_max_len else test
        for _ in range(48):
            bits = [rng.choice(["mark", "skip"]) for _ in range(length)]
            label = f"mod_{bits.count('mark') % 4}"
            target.append(ClassificationExample(tuple(["count", *bits]), label))
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def make_next_token_task(
    seed: int = 7,
    train_max_len: int = 8,
    test_max_len: int = 16,
) -> tuple[list[ClassificationExample], list[ClassificationExample]]:
    rng = random.Random(seed)
    motifs = [
        ("a", "b", "c"),
        ("x", "y"),
        ("up", "down", "stay"),
        ("left", "right"),
    ]
    train: list[ClassificationExample] = []
    test: list[ClassificationExample] = []
    for motif in motifs:
        for length in range(2, test_max_len + 1):
            for offset in range(len(motif)):
                seq = tuple(motif[(offset + idx) % len(motif)] for idx in range(length))
                label = motif[(offset + length) % len(motif)]
                target = train if length <= train_max_len else test
                target.append(ClassificationExample(("prefix", *seq, "next"), label))
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def make_scientific_surrogate_task(seed: int = 7) -> tuple[list[ClassificationExample], list[ClassificationExample]]:
    """Tiny diffusion-surrogate task for the scientific-computing roadmap slice.

    Inputs describe a discretized diffusion coefficient and time. The label is a
    bucketed value of exp(-k*t), mimicking a coarse one-step PDE surrogate.
    """

    rng = random.Random(seed)
    train: list[ClassificationExample] = []
    test: list[ClassificationExample] = []
    for k_idx in range(1, 10):
        for t_idx in range(1, 10):
            value = pow(2.718281828, -(k_idx / 10.0) * (t_idx / 4.0))
            if value > 0.75:
                label = "high"
            elif value > 0.45:
                label = "medium"
            else:
                label = "low"
            tokens = ("diffuse", f"k{k_idx}", f"t{t_idx}", "bucket")
            target = test if (k_idx + t_idx) % 4 == 0 else train
            target.append(ClassificationExample(tokens, label))
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def make_clevr_binding_task(seed: int = 7) -> tuple[list[ClassificationExample], list[ClassificationExample]]:
    """CLEVR-style attribute binding with more objects at test time."""

    colors = ["red", "blue", "green", "yellow", "purple"]
    shapes = ["circle", "square", "triangle", "star"]
    sizes = ["small", "large"]
    materials = ["rubber", "metal"]
    rng = random.Random(seed)
    train: list[ClassificationExample] = []
    test: list[ClassificationExample] = []

    def make_scene(n_objects: int, query_index: int) -> ClassificationExample:
        specs = []
        used_colors = rng.sample(colors, n_objects)
        for idx, color in enumerate(used_colors):
            specs.append((color, rng.choice(shapes), rng.choice(sizes), rng.choice(materials)))
        query_color, query_shape, _, _ = specs[query_index]
        tokens: list[str] = ["scene"]
        for color, shape, size, material in specs:
            tokens.extend(["obj", color, shape, size, material])
        tokens.extend(["query", query_color, "shape"])
        return ClassificationExample(tuple(tokens), query_shape)

    for _ in range(384):
        n_objects = rng.choice([2, 3])
        train.append(make_scene(n_objects, rng.randrange(n_objects)))
    for _ in range(192):
        n_objects = rng.choice([4, 5])
        test.append(make_scene(n_objects, rng.randrange(n_objects)))
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def make_arithmetic_task(
    seed: int = 7,
    train_max_terms: int = 4,
    test_max_terms: int = 8,
) -> tuple[list[ClassificationExample], list[ClassificationExample]]:
    """Symbolic arithmetic length-generalization task with exact mod labels."""

    rng = random.Random(seed)
    digits = list(range(10))
    train: list[ClassificationExample] = []
    test: list[ClassificationExample] = []
    for n_terms in range(2, test_max_terms + 1):
        target = train if n_terms <= train_max_terms else test
        for _ in range(80):
            values = [rng.choice(digits) for _ in range(n_terms)]
            ops = [rng.choice(["plus", "minus"]) for _ in range(n_terms - 1)]
            total = values[0]
            tokens = ["calc", f"n{values[0]}"]
            for op, value in zip(ops, values[1:]):
                tokens.extend([op, f"n{value}"])
                total = total + value if op == "plus" else total - value
            target.append(ClassificationExample(tuple(tokens), f"mod_{total % 10}"))
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def make_adaptive_compute_task(seed: int = 7) -> tuple[list[ClassificationExample], list[ClassificationExample]]:
    """Easy/hard mixture for compute-proportional reasoning diagnostics."""

    rng = random.Random(seed)
    train: list[ClassificationExample] = []
    test: list[ClassificationExample] = []
    symbols = ["a", "b", "c", "d", "e", "f"]
    symbol_value = {symbol: idx for idx, symbol in enumerate(symbols)}

    for split, target, count in [("train", train, 512), ("test", test, 256)]:
        for _ in range(count):
            hard = rng.random() < (0.45 if split == "train" else 0.55)
            if not hard:
                symbol = rng.choice(symbols)
                tokens = ("easy", "lookup", symbol)
                label = f"mod_{symbol_value[symbol] % 3}"
            else:
                length = rng.randint(5, 10 if split == "train" else 14)
                seq = [rng.choice(symbols) for _ in range(length)]
                value = sum((idx + 1) * symbol_value[symbol] for idx, symbol in enumerate(seq))
                tokens = tuple(["hard", "compose", *seq, "reduce"])
                label = f"mod_{value % 3}"
            target.append(ClassificationExample(tokens, label))
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


TASK_BUILDERS = {
    "adaptive_compute": make_adaptive_compute_task,
    "arithmetic": make_arithmetic_task,
    "binding": make_binding_task,
    "clevr_binding": make_clevr_binding_task,
    "length": make_length_generalization_task,
    "next_token": make_next_token_task,
    "scientific": make_scientific_surrogate_task,
}
