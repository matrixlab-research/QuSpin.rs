from __future__ import annotations

from fractions import Fraction
from functools import cached_property
from typing import Any

import numpy as np

from qmbed._ffi import command


def _reject_options(family: str, options: dict[str, Any]) -> None:
    unsupported = {
        name: value
        for name, value in options.items()
        if value is not None and name != "a"
    }
    if options.get("a", 1) != 1:
        unsupported["a"] = options["a"]
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise NotImplementedError(f"{family} does not support these blocks yet: {names}")


def _spin_twice(spin: str | int | float) -> int:
    value = Fraction(str(spin))
    doubled = value * 2
    if doubled.denominator != 1 or doubled <= 0:
        raise ValueError(f"invalid spin quantum number {spin!r}")
    return int(doubled)


class _PackedBasis:
    _request: dict[str, Any]
    N: int

    @cached_property
    def _description(self) -> dict[str, Any]:
        return command({"operation": "describe_basis", "basis": self._request})

    @property
    def Ns(self) -> int:
        return int(self._description["dimension"])

    @property
    def states(self) -> np.ndarray:
        return np.asarray(
            [int(state) for state in self._description["states"]],
            dtype=object,
        )

    def __len__(self) -> int:
        return self.Ns

    @property
    def _site_permutation(self) -> list[int]:
        return list(range(self.N - 1, -1, -1))

    def expanded_form(self, static, dynamic):
        return static, dynamic


class spin_basis_1d(_PackedBasis):
    def __init__(
        self,
        L: int,
        Nup: int | None = None,
        m: float | None = None,
        S: str | int | float = "1/2",
        pauli: bool | int = True,
        kblock: int | None = None,
        pblock: int | None = None,
        a: int = 1,
        **blocks,
    ):
        if m is not None:
            if Nup is not None:
                raise ValueError("Nup and m cannot both be specified")
            Nup = round((float(m) + 0.5) * L)
        _reject_options("spin_basis_1d", {"a": a, **blocks})
        self.N = int(L)
        self._request = {
            "kind": "spin",
            "sites": self.N,
            "spin_twice": _spin_twice(S),
            "up": Nup,
            "momentum": kblock,
            "parity": pblock,
            "pauli": bool(pauli),
            "reverse": True,
        }


class boson_basis_1d(_PackedBasis):
    def __init__(
        self,
        L: int,
        Nb: int | None = None,
        sps: int | None = None,
        kblock: int | None = None,
        pblock: int | None = None,
        a: int = 1,
        **blocks,
    ):
        _reject_options(
            "boson_basis_1d",
            {"kblock": kblock, "pblock": pblock, "a": a, **blocks},
        )
        self.N = int(L)
        states_per_site = int(sps if sps is not None else (Nb + 1 if Nb is not None else 2))
        self._request = {
            "kind": "boson",
            "sites": self.N,
            "particles": Nb,
            "states_per_site": states_per_site,
            "reverse": True,
        }


class ho_basis(boson_basis_1d):
    def __init__(self, N: int):
        super().__init__(1, sps=int(N) + 1)


class spinless_fermion_basis_1d(_PackedBasis):
    def __init__(
        self,
        L: int,
        Nf: int | None = None,
        kblock: int | None = None,
        pblock: int | None = None,
        a: int = 1,
        **blocks,
    ):
        _reject_options(
            "spinless_fermion_basis_1d",
            {"pblock": pblock, "a": a, **blocks},
        )
        self.N = int(L)
        self._request = {
            "kind": "spinless_fermion",
            "sites": self.N,
            "particles": Nf,
            "momentum": kblock,
            "reverse": True,
        }


class spinful_fermion_basis_1d(_PackedBasis):
    def __init__(
        self,
        L: int,
        Nf: tuple[int, int] | None = None,
        **blocks,
    ):
        _reject_options("spinful_fermion_basis_1d", blocks)
        self.N = int(L)
        particles_up, particles_down = (None, None) if Nf is None else Nf
        self._request = {
            "kind": "spinful_fermion",
            "sites": self.N,
            "particles_up": particles_up,
            "particles_down": particles_down,
            "reverse": True,
        }


__all__ = [
    "boson_basis_1d",
    "ho_basis",
    "spin_basis_1d",
    "spinful_fermion_basis_1d",
    "spinless_fermion_basis_1d",
]
