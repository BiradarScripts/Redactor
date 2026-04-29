import unittest

import torch

from mcn.clifford import CliffordAlgebra


class CliffordAlgebraTests(unittest.TestCase):
    def test_euclidean_basis_products(self):
        algebra = CliffordAlgebra(3)
        one = torch.zeros(8)
        e1 = torch.zeros(8)
        e2 = torch.zeros(8)
        one[0] = 1
        e1[1] = 1
        e2[2] = 1

        self.assertTrue(torch.allclose(algebra.geometric_product(one, e1), e1))
        self.assertTrue(torch.allclose(algebra.geometric_product(e1, e1), one))

        e12 = torch.zeros(8)
        e12[3] = 1
        self.assertTrue(torch.allclose(algebra.geometric_product(e1, e2), e12))
        self.assertTrue(torch.allclose(algebra.geometric_product(e2, e1), -e12))


if __name__ == "__main__":
    unittest.main()
