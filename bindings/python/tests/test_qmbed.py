import unittest

import numpy as np
import qmbed
from qmbed.compat import quspin
from qmbed._ffi import QmbedError, command
from quspin.basis import basis_int_to_python_int, spin_basis_1d, spin_basis_general
from quspin.operators import hamiltonian
from quspin.operators._make_hamiltonian import _consolidate_static


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

    def test_python_site_maps_reproduce_the_optimized_momentum_sector(self):
        sites = 6
        translation = (np.arange(sites) + 1) % sites
        general = spin_basis_general(
            sites,
            Nup=3,
            pauli=False,
            translation=(translation, 1),
        )
        optimized = spin_basis_1d(
            sites,
            Nup=3,
            pauli=False,
            kblock=1,
        )

        np.testing.assert_array_equal(general.states, optimized.states)
        static = [
            [
                "+-",
                [[1.0j, site, (site + 1) % sites] for site in range(sites)],
            ],
            [
                "-+",
                [[-1.0j, site, (site + 1) % sites] for site in range(sites)],
            ],
        ]
        general_operator = hamiltonian(
            static,
            [],
            basis=general,
            check_herm=False,
            check_pcon=False,
            check_symm=False,
        )
        optimized_operator = hamiltonian(
            static,
            [],
            basis=optimized,
            check_herm=False,
            check_pcon=False,
            check_symm=False,
        )
        np.testing.assert_allclose(
            general_operator.toarray(),
            optimized_operator.toarray(),
            atol=1.0e-12,
        )

    def test_low_level_basis_operations_share_one_rust_action_protocol(self):
        basis = spin_basis_general(2, pauli=-1)
        static = [["y", [[0.75, 0]]], ["+", [[-0.5j, 1]]]]
        op_list = _consolidate_static(static)
        operator = hamiltonian(
            static,
            [],
            basis=basis,
            check_herm=False,
            check_pcon=False,
            check_symm=False,
        ).toarray()
        vector = np.asarray([1 + 0.5j, -2j, 0.25, -0.5 + 0.75j])

        actions = [
            (False, False, operator),
            (True, False, operator.T),
            (False, True, operator.conj()),
            (True, True, operator.conj().T),
        ]
        for transposed, conjugated, matrix in actions:
            actual = basis.inplace_Op(
                vector,
                op_list,
                np.complex128,
                transposed=transposed,
                conjugated=conjugated,
            )
            np.testing.assert_allclose(actual, matrix.dot(vector), atol=1.0e-12)

        initial = np.ones_like(vector)
        returned = basis.inplace_Op(
            vector,
            op_list,
            np.complex128,
            v_out=initial,
        )
        self.assertIs(returned, initial)
        np.testing.assert_allclose(returned, 1.0 + operator.dot(vector), atol=1.0e-12)

        elements, rows, columns = basis.Op("y", [0], 0.75, np.complex128)
        reconstructed = np.zeros_like(operator)
        reconstructed[rows, columns] = elements
        expected = hamiltonian(
            [["y", [[0.75, 0]]]],
            [],
            basis=basis,
            check_herm=False,
            check_pcon=False,
            check_symm=False,
        ).toarray()
        np.testing.assert_allclose(reconstructed, expected, atol=1.0e-12)

        elements, bras, kets = basis.Op_bra_ket(
            "+",
            [0],
            1.5,
            np.float64,
            basis.states,
        )
        self.assertTrue(np.all(elements == 1.5))
        self.assertTrue(np.all(bras > kets))
        self.assertTrue(all(basis_int_to_python_int(value) == int(value) for value in bras))

    def test_python_pauli_modes_map_to_distinct_rust_normalizations(self):
        spin = spin_basis_1d(1, pauli=0)
        pauli = spin_basis_1d(1, pauli=1)
        cartesian = spin_basis_1d(1, pauli=-1)

        spin_raising = spin.Op("+", [0], 1.0, np.float64)[0]
        pauli_raising = pauli.Op("+", [0], 1.0, np.float64)[0]
        cartesian_raising = cartesian.Op("+", [0], 1.0, np.float64)[0]
        np.testing.assert_allclose(pauli_raising, 2.0 * spin_raising)
        np.testing.assert_allclose(cartesian_raising, spin_raising)

        spin_x = spin.Op("x", [0], 1.0, np.float64)[0]
        pauli_x = pauli.Op("x", [0], 1.0, np.float64)[0]
        cartesian_x = cartesian.Op("x", [0], 1.0, np.float64)[0]
        np.testing.assert_allclose(pauli_x, 2.0 * spin_x)
        np.testing.assert_allclose(cartesian_x, 2.0 * spin_x)


if __name__ == "__main__":
    unittest.main()
