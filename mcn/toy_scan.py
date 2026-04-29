from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

import torch


PRIMITIVES = {
    "walk": ("WALK",),
    "run": ("RUN",),
    "jump": ("JUMP",),
    "look": ("LOOK",),
    "turn left": ("LTURN",),
    "turn right": ("RTURN",),
}


@dataclass(frozen=True)
class ToyScanExample:
    command: str
    actions: tuple[str, ...]


class Vocab:
    def __init__(self, tokens: Iterable[str]):
        specials = ["<pad>", "<bos>", "<eos>", "<unk>"]
        ordered = specials + sorted(set(tokens) - set(specials))
        self.stoi = {token: i for i, token in enumerate(ordered)}
        self.itos = ordered

    @property
    def pad_id(self) -> int:
        return self.stoi["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.stoi["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.stoi["<eos>"]

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: Iterable[str], add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = [self.stoi["<bos>"]] if add_bos else []
        ids.extend(self.stoi.get(token, self.stoi["<unk>"]) for token in tokens)
        if add_eos:
            ids.append(self.stoi["<eos>"])
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> list[str]:
        out = []
        for idx in ids:
            token = self.itos[int(idx)]
            if skip_special and token.startswith("<"):
                continue
            out.append(token)
        return out


def _expand(command: str, actions: tuple[str, ...], depth: int) -> set[ToyScanExample]:
    examples = {ToyScanExample(command, actions)}
    if depth <= 0:
        return examples

    examples.add(ToyScanExample(f"{command} twice", actions * 2))
    examples.add(ToyScanExample(f"{command} thrice", actions * 3))

    for other_command, other_actions in PRIMITIVES.items():
        examples.add(ToyScanExample(f"{command} and {other_command}", actions + other_actions))
        examples.add(ToyScanExample(f"{command} after {other_command}", other_actions + actions))
    return examples


def generate_toy_scan(max_depth: int = 2, max_action_len: int = 12) -> list[ToyScanExample]:
    examples: set[ToyScanExample] = set()
    frontier = [ToyScanExample(command, actions) for command, actions in PRIMITIVES.items()]

    for depth in range(max_depth + 1):
        next_frontier = []
        for example in frontier:
            if len(example.actions) <= max_action_len:
                examples.add(example)
            if depth < max_depth:
                expanded = _expand(example.command, example.actions, 1)
                for item in expanded:
                    if len(item.actions) <= max_action_len:
                        next_frontier.append(item)
        frontier = next_frontier

    return sorted(examples, key=lambda item: (len(item.command.split()), item.command))


def make_jump_composition_split(seed: int = 7) -> tuple[list[ToyScanExample], list[ToyScanExample]]:
    examples = generate_toy_scan()
    train = []
    test = []
    for example in examples:
        command_tokens = example.command.split()
        is_jump = "jump" in command_tokens
        is_primitive_jump = example.command == "jump"
        if is_jump and not is_primitive_jump:
            test.append(example)
        else:
            train.append(example)

    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def make_random_split(
    seed: int = 7,
    test_fraction: float = 0.2,
) -> tuple[list[ToyScanExample], list[ToyScanExample]]:
    examples = generate_toy_scan()
    rng = random.Random(seed)
    rng.shuffle(examples)
    test_size = max(1, int(len(examples) * test_fraction))
    return examples[test_size:], examples[:test_size]


def build_vocabs(examples: Iterable[ToyScanExample]) -> tuple[Vocab, Vocab]:
    src_tokens = []
    tgt_tokens = []
    for example in examples:
        src_tokens.extend(example.command.split())
        tgt_tokens.extend(example.actions)
    return Vocab(src_tokens), Vocab(tgt_tokens)


def collate_examples(
    examples: list[ToyScanExample],
    src_vocab: Vocab,
    tgt_vocab: Vocab,
) -> dict[str, torch.Tensor]:
    src_ids = [src_vocab.encode(example.command.split(), add_eos=True) for example in examples]
    tgt_ids = [tgt_vocab.encode(example.actions, add_bos=True, add_eos=True) for example in examples]
    src_max = max(len(ids) for ids in src_ids)
    tgt_max = max(len(ids) for ids in tgt_ids)

    src = torch.full((len(examples), src_max), src_vocab.pad_id, dtype=torch.long)
    tgt = torch.full((len(examples), tgt_max), tgt_vocab.pad_id, dtype=torch.long)
    for i, ids in enumerate(src_ids):
        src[i, : len(ids)] = torch.tensor(ids)
    for i, ids in enumerate(tgt_ids):
        tgt[i, : len(ids)] = torch.tensor(ids)

    return {
        "src": src,
        "tgt_in": tgt[:, :-1],
        "tgt_out": tgt[:, 1:],
    }
