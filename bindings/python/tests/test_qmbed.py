import unittest

import qmbed
from qmbed.compat import quspin


class QmbedBindingTests(unittest.TestCase):
    def test_native_and_compatibility_paths_share_the_rust_solver(self):
        basis = qmbed.SpinBasis(2)
        coupling = lambda value: qmbed.Coupling(value, (0, 1))
        terms = (
            qmbed.OperatorSpec(
                qmbed.OpProduct((qmbed.LocalOperator.Z, qmbed.LocalOperator.Z)),
                (coupling(1.0),),
            ),
            quspin.operator_term("+-", (coupling(0.5),)),
            quspin.operator_term("-+", (coupling(0.5),)),
        )
        result = qmbed.eigsh(basis, terms, qmbed.EigshOptions(2))
        self.assertEqual(result.dimension, 4)
        self.assertAlmostEqual(result.eigenvalues[0], -0.75, places=10)
        self.assertTrue(result.converged)

    def test_quspin_static_adapter(self):
        result = quspin.eigsh(
            qmbed.SpinBasis(2),
            [
                ("zz", [[1.0, 0, 1]]),
                ("+-", [[0.5, 0, 1]]),
                ("-+", [[0.5, 0, 1]]),
            ],
            k=2,
            which="SA",
        )
        self.assertAlmostEqual(result.eigenvalues[0], -0.75, places=10)


if __name__ == "__main__":
    unittest.main()
