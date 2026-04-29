from __future__ import annotations

import functools

import torch
from torch import nn


@functools.lru_cache(maxsize=16)
def _multiplication_tables(p: int) -> tuple[torch.Tensor, torch.Tensor]:
    if p <= 0:
        raise ValueError("p must be positive")

    n_blades = 1 << p
    index = torch.empty(n_blades, n_blades, dtype=torch.long)
    sign = torch.empty(n_blades, n_blades, dtype=torch.float32)

    for left in range(n_blades):
        for right in range(n_blades):
            swaps = 0
            for bit in range(p):
                if left & (1 << bit):
                    swaps += (right & ((1 << bit) - 1)).bit_count()
            index[left, right] = left ^ right
            sign[left, right] = -1.0 if swaps % 2 else 1.0

    return index, sign


class CliffordAlgebra(nn.Module):
    """Euclidean Clifford algebra Cl(p, 0) with dense multivector tensors.

    Multivectors are stored as tensors whose final dimension is ``2 ** p``.
    Basis blade 0 is the scalar grade. Basis blade bitmasks follow the usual
    exterior basis convention, so e1 is mask 1, e2 is mask 2, and e1e2 is mask 3.
    """

    def __init__(self, p: int):
        super().__init__()
        index, sign = _multiplication_tables(p)
        self.p = p
        self.n_blades = 1 << p
        self.register_buffer("product_index", index, persistent=False)
        self.register_buffer("product_sign", sign, persistent=False)

    def geometric_product(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if left.shape[-1] != self.n_blades or right.shape[-1] != self.n_blades:
            raise ValueError(f"expected final dimension {self.n_blades}")

        products = left.unsqueeze(-1) * right.unsqueeze(-2) * self.product_sign
        out = left.new_zeros(*products.shape[:-2], self.n_blades)
        flat_index = self.product_index.reshape(-1).expand(*products.shape[:-2], -1)
        out.scatter_add_(-1, flat_index, products.reshape(*products.shape[:-2], -1))
        return out

    def scalar_part(self, multivector: torch.Tensor) -> torch.Tensor:
        return multivector[..., 0]

    def grade_mask(self, grade: int, device: torch.device | None = None) -> torch.Tensor:
        if grade < 0 or grade > self.p:
            raise ValueError("grade must be between 0 and p")
        values = [1.0 if blade.bit_count() == grade else 0.0 for blade in range(self.n_blades)]
        return torch.tensor(values, dtype=torch.float32, device=device)

    def project_grade(self, multivector: torch.Tensor, grade: int) -> torch.Tensor:
        return multivector * self.grade_mask(grade, multivector.device)
