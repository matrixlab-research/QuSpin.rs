import unittest

import numpy as np
import qmbed
from qmbed.compat import quspin
from qmbed._ffi import QmbedError, command
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian


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

    def test_quspin_hamiltonian_reuses_and_releases_one_native_model(self):
        basis = spin_basis_1d(2)
        operator = hamiltonian(
            [["zz", [[1.0, 0, 1]]]],
            [],
            basis=basis,
            dtype=np.float64,
        )
        handle = operator._model.handle

        matrix = operator.toarray()
        eigenvalues = operator.eigvalsh()
        description = command({"operation": "describe_model", "handle": handle})

        self.assertEqual(matrix.shape, (4, 4))
        self.assertEqual(len(eigenvalues), 4)
        self.assertEqual(description["dimension"], 4)
        self.assertEqual(operator._model.handle, handle)

        operator.close()
        self.assertTrue(operator.closed)
        operator.close()
        with self.assertRaisesRegex(QmbedError, "model is closed"):
            operator.toarray()
        with self.assertRaisesRegex(QmbedError, "is not registered"):
            command({"operation": "describe_model", "handle": handle})


if __name__ == "__main__":
    unittest.main()
