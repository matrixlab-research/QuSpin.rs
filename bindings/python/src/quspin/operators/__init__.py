from __future__ import annotations

from typing import Any

import numpy as np

from qmbed._ffi import NativeModel
from qmbed.compat.quspin import terms_from_static


_TARGETS = {
    "SA": "smallest_algebraic",
    "LA": "largest_algebraic",
    "SM": "smallest_magnitude",
    "LM": "largest_magnitude",
    "BE": "both_ends",
}


class _OperatorView:
    def __init__(self, owner, *, transposed=False, conjugated=False):
        self._owner = owner
        self._transposed = bool(transposed)
        self._conjugated = bool(conjugated)

    @property
    def shape(self):
        return self._owner.shape[::-1] if self._transposed else self._owner.shape

    @property
    def T(self):
        return _OperatorView(
            self._owner,
            transposed=not self._transposed,
            conjugated=self._conjugated,
        )

    @property
    def H(self):
        return _OperatorView(
            self._owner,
            transposed=not self._transposed,
            conjugated=not self._conjugated,
        )

    def conj(self):
        return _OperatorView(
            self._owner,
            transposed=self._transposed,
            conjugated=not self._conjugated,
        )

    conjugate = conj

    def dot(self, vector):
        return self._owner._dot_action(
            vector,
            transposed=self._transposed,
            conjugated=self._conjugated,
        )


class hamiltonian:
    def __init__(
        self,
        static,
        dynamic,
        *,
        basis=None,
        N: int | None = None,
        Nup: int | None = None,
        S: str | int | float = "1/2",
        pauli: bool | int = True,
        dtype=np.complex128,
        check_herm: bool = True,
        check_pcon: bool = True,
        check_symm: bool = True,
        **basis_options,
    ):
        if dynamic:
            raise NotImplementedError("dynamic Hamiltonian compatibility is not implemented yet")
        if basis is None:
            if N is None:
                raise ValueError("basis or N must be supplied")
            from quspin.basis import spin_basis_1d

            basis = spin_basis_1d(
                N,
                Nup=Nup,
                S=S,
                pauli=pauli,
                **basis_options,
            )
        elif N is not None or Nup is not None or basis_options:
            raise ValueError("basis construction options cannot accompany an explicit basis")

        self.basis = basis
        self.dtype = np.dtype(dtype)
        self._terms = tuple(terms_from_static(static))
        self._checks = {
            "hermiticity": bool(check_herm),
            "particle_conservation": bool(check_pcon),
            "symmetry_compatibility": bool(check_symm),
        }
        self._model = NativeModel(
            {
                "basis": self.basis._request,
                "terms": [term.request() for term in self._terms],
                "site_permutation": self.basis._site_permutation,
                "checks": self._checks,
            }
        )
        self.Ns = self._model.dimension

    @property
    def shape(self) -> tuple[int, int]:
        return self.Ns, self.Ns

    @property
    def get_shape(self) -> tuple[int, int]:
        return self.shape

    @property
    def closed(self) -> bool:
        return self._model.closed

    def close(self) -> None:
        self._model.close()

    def __enter__(self) -> hamiltonian:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def _execute(self, operation: str, **options: Any) -> dict[str, Any]:
        return self._model.execute(operation, **options)

    def _coerce_matrix(self, result: dict[str, Any]) -> np.ndarray:
        rows, columns = result["shape"]
        matrix = np.zeros((rows, columns), dtype=np.complex128)
        for entry in result["entries"]:
            matrix[entry["row"], entry["column"]] = complex(*entry["value"])
        if self.dtype.kind != "c":
            if np.any(np.abs(matrix.imag) > 1.0e-12):
                raise TypeError("complex operator cannot be represented by a real dtype")
            matrix = matrix.real
        return np.asarray(matrix, dtype=self.dtype)

    def _dot_action(self, vector, *, transposed=False, conjugated=False):
        input_array = np.asanyarray(vector)
        if input_array.ndim == 0 or input_array.shape[0] != self.Ns:
            raise ValueError("dimension mismatch")
        result_dtype = np.result_type(input_array.dtype, self.dtype)
        input_array = input_array.astype(result_dtype, order="C", copy=False)
        input_matrix = input_array.reshape((self.Ns, -1))
        vectors = [
            [[complex(value).real, complex(value).imag] for value in input_matrix[:, column]]
            for column in range(input_matrix.shape[1])
        ]
        if transposed and conjugated:
            action = "adjoint"
        elif transposed:
            action = "transpose"
        elif conjugated:
            action = "conjugate"
        else:
            action = "normal"
        result = self._execute("apply_model", vectors=vectors, action=action)
        applied = np.column_stack(
            [
                np.asarray([complex(*value) for value in output])
                for output in result["vectors"]
            ]
        ).reshape(input_array.shape)
        if np.dtype(result_dtype).kind != "c":
            tolerance = 10 * np.finfo(np.float64).eps
            if np.any(np.abs(applied.imag) > tolerance):
                raise TypeError("complex result cannot be represented by a real dtype")
            applied = applied.real
        return np.asarray(applied, dtype=result_dtype)

    def toarray(self, time: float | None = None) -> np.ndarray:
        if time is not None:
            raise NotImplementedError("time-dependent materialization is not implemented yet")
        return self._coerce_matrix(self._execute("materialize_model", format="csc"))

    def todense(self, time: float | None = None) -> np.matrix:
        return np.asmatrix(self.toarray(time))

    def eigvalsh(self, time: float | None = None) -> np.ndarray:
        if time is not None:
            raise NotImplementedError("time-dependent eigvalsh is not implemented yet")
        result = self._execute("eigh_model", eigenvectors=False)
        return np.asarray(result["eigenvalues"])

    def eigh(self, time: float | None = None):
        if time is not None:
            raise NotImplementedError("time-dependent eigh is not implemented yet")
        result = self._execute("eigh_model", eigenvectors=True)
        vectors = np.column_stack(
            [
                np.asarray([complex(*value) for value in vector])
                for vector in result["eigenvectors"]
            ]
        )
        return np.asarray(result["eigenvalues"]), vectors

    def eigsh(
        self,
        *,
        k: int,
        which: str = "SA",
        sigma: float | None = None,
        return_eigenvectors: bool = True,
        maxiter: int = 1_000,
        tol: float = 1.0e-10,
        ncv: int | None = None,
        v0=None,
        **_options,
    ):
        if v0 is not None:
            raise NotImplementedError("eigsh initial vectors are not exposed yet")
        target = (
            {"kind": "shift", "value": float(sigma)}
            if sigma is not None
            else {"kind": _TARGETS[which]}
        )
        result = self._execute(
            "eigsh_model",
            format="csc",
            solver={
                "eigenpairs": int(k),
                "target": target,
                "krylov_dimension": ncv,
                "tolerance": float(tol),
                "max_iterations": int(maxiter),
                "eigenvectors": bool(return_eigenvectors),
            },
        )
        values = np.asarray(result["eigenvalues"])
        if not return_eigenvectors:
            return values
        vectors = np.column_stack(
            [
                np.asarray([complex(*value) for value in vector])
                for vector in result["eigenvectors"]
            ]
        )
        return values, vectors

    def dot(self, vector):
        return self._dot_action(vector)

    @property
    def T(self):
        return _OperatorView(self, transposed=True)

    @property
    def H(self):
        return _OperatorView(self, transposed=True, conjugated=True)

    def conj(self):
        return _OperatorView(self, conjugated=True)

    conjugate = conj

    def trace(self):
        return np.trace(self.toarray())


class quantum_LinearOperator(hamiltonian):
    """Matrix-free QuSpin-compatible view over one persistent Rust model."""

    def __init__(self, static, **options):
        super().__init__(static, [], **options)


__all__ = ["hamiltonian", "quantum_LinearOperator"]
