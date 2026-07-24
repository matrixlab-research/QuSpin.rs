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


def _rust_site_map(site_map, sites: int) -> tuple[list[int], list[bool]]:
    values = [int(value) for value in np.asarray(site_map).reshape(-1)]
    if len(values) != sites:
        raise ValueError(f"symmetry map has {len(values)} sites, expected {sites}")
    destinations = [0] * sites
    inverted = [False] * sites
    for python_source, encoded_destination in enumerate(values):
        is_inverted = encoded_destination < 0
        python_destination = (
            -encoded_destination - 1 if is_inverted else encoded_destination
        )
        if not 0 <= python_destination < sites:
            raise ValueError("symmetry map contains an out-of-range site")
        rust_source = sites - python_source - 1
        destinations[rust_source] = sites - python_destination - 1
        inverted[rust_source] = is_inverted
    if len(set(destinations)) != sites:
        raise ValueError("symmetry site map must be bijective")
    return destinations, inverted


def _symmetry_request(
    site_map,
    sector: int,
    *,
    sites: int,
    states_per_site: int,
    fermionic: bool = False,
) -> dict[str, Any]:
    destinations, inverted = _rust_site_map(site_map, sites)
    request: dict[str, Any] = {
        "destinations": destinations,
        "sector": int(sector),
    }
    if any(inverted):
        if fermionic:
            raise NotImplementedError(
                "fermionic particle-hole maps require an explicit phase convention"
            )
        identity = list(range(states_per_site))
        reversed_digits = list(reversed(identity))
        request["local_permutations"] = [
            reversed_digits if flip else identity for flip in inverted
        ]
    return request


def _general_symmetries(
    blocks: dict[str, Any],
    *,
    sites: int,
    states_per_site: int,
    fermionic: bool = False,
) -> list[dict[str, Any]]:
    symmetries = []
    for name, block in blocks.items():
        if block is None:
            continue
        if not isinstance(block, (tuple, list)) or len(block) != 2:
            raise ValueError(f"{name} must be a (site_map, sector) pair")
        site_map, sector = block
        symmetries.append(
            _symmetry_request(
                site_map,
                sector,
                sites=sites,
                states_per_site=states_per_site,
                fermionic=fermionic,
            )
        )
    return symmetries


def _one_dimensional_symmetries(
    sites: int,
    *,
    states_per_site: int,
    momentum: int | None,
    parity: int | None,
) -> list[dict[str, Any]]:
    blocks: dict[str, Any] = {}
    if momentum is not None:
        translation = (np.arange(sites) + 1) % sites
        blocks["translation"] = (translation, int(momentum))
    if parity is not None:
        if parity not in (-1, 1):
            raise ValueError("pblock must be either -1 or +1")
        if momentum is not None:
            normalized = int(momentum) % sites
            if normalized != 0 and 2 * normalized != sites:
                raise ValueError("parity can accompany momentum only at k=0 or k=pi")
        reflection = np.arange(sites)[::-1]
        blocks["parity"] = (reflection, 0 if parity == 1 else 1)
    return _general_symmetries(
        blocks,
        sites=sites,
        states_per_site=states_per_site,
    )


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
        spin_twice = _spin_twice(S)
        if m is not None:
            if Nup is not None:
                raise ValueError("Nup and m cannot both be specified")
            Nup = round((float(m) + spin_twice / 2) * L)
        _reject_options("spin_basis_1d", {"a": a, **blocks})
        self.N = int(L)
        self._request = {
            "kind": "spin",
            "sites": self.N,
            "spin_twice": spin_twice,
            "up": Nup,
            "momentum": None if kblock is None else -int(kblock),
            "parity": pblock,
            "pauli": bool(pauli) if spin_twice == 1 else False,
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
        _reject_options("boson_basis_1d", {"a": a, **blocks})
        self.N = int(L)
        states_per_site = int(
            sps if sps is not None else (Nb + 1 if Nb is not None else 2)
        )
        self._request = {
            "kind": "boson",
            "sites": self.N,
            "particles": Nb,
            "states_per_site": states_per_site,
            "symmetries": _one_dimensional_symmetries(
                self.N,
                states_per_site=states_per_site,
                momentum=kblock,
                parity=pblock,
            ),
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
            "momentum": None if kblock is None else -int(kblock),
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


class spin_basis_general(_PackedBasis):
    def __init__(
        self,
        N: int,
        Nup: int | None = None,
        m: float | None = None,
        S: str | int | float = "1/2",
        pauli: bool | int = True,
        make_basis: bool = True,
        block_order=None,
        **blocks,
    ):
        spin_twice = _spin_twice(S)
        if not make_basis:
            raise NotImplementedError("deferred general-basis construction is not implemented")
        if block_order is not None:
            ordered = {
                name: blocks.pop(name)
                for name in block_order
                if name in blocks
            }
            ordered.update(blocks)
            blocks = ordered
        if m is not None:
            if Nup is not None:
                raise ValueError("Nup and m cannot both be specified")
            Nup = round((float(m) + spin_twice / 2) * N)
        if Nup is not None and not isinstance(Nup, (int, np.integer)):
            raise NotImplementedError("unions of spin particle sectors are not implemented")
        self.N = int(N)
        self._request = {
            "kind": "spin",
            "sites": self.N,
            "spin_twice": spin_twice,
            "up": None if Nup is None else int(Nup),
            "momentum": None,
            "parity": None,
            "pauli": bool(pauli) if spin_twice == 1 else False,
            "symmetries": _general_symmetries(
                blocks,
                sites=self.N,
                states_per_site=spin_twice + 1,
            ),
            "reverse": True,
        }


class boson_basis_general(_PackedBasis):
    def __init__(
        self,
        N: int,
        Nb: int | None = None,
        sps: int | None = None,
        **blocks,
    ):
        self.N = int(N)
        states_per_site = int(
            sps if sps is not None else (Nb + 1 if Nb is not None else 2)
        )
        self._request = {
            "kind": "boson",
            "sites": self.N,
            "particles": Nb,
            "states_per_site": states_per_site,
            "symmetries": _general_symmetries(
                blocks,
                sites=self.N,
                states_per_site=states_per_site,
            ),
            "reverse": True,
        }


class spinless_fermion_basis_general(_PackedBasis):
    def __init__(
        self,
        N: int,
        Nf: int | None = None,
        **blocks,
    ):
        if Nf is not None and not isinstance(Nf, (int, np.integer)):
            raise NotImplementedError("unions of fermion particle sectors are not implemented")
        self.N = int(N)
        self._request = {
            "kind": "spinless_fermion",
            "sites": self.N,
            "particles": None if Nf is None else int(Nf),
            "momentum": None,
            "symmetries": _general_symmetries(
                blocks,
                sites=self.N,
                states_per_site=2,
                fermionic=True,
            ),
            "reverse": True,
        }


class spinful_fermion_basis_general(_PackedBasis):
    def __init__(
        self,
        N: int,
        Nf: tuple[int, int] | None = None,
        **blocks,
    ):
        self.N = int(N)
        particles_up, particles_down = (None, None) if Nf is None else Nf
        spatial_symmetries = _general_symmetries(
            blocks,
            sites=self.N,
            states_per_site=2,
            fermionic=True,
        )
        symmetries = []
        for symmetry in spatial_symmetries:
            destinations = symmetry["destinations"]
            symmetries.append(
                {
                    **symmetry,
                    "destinations": destinations
                    + [self.N + destination for destination in destinations],
                }
            )
        self._request = {
            "kind": "spinful_fermion",
            "sites": self.N,
            "particles_up": None if particles_up is None else int(particles_up),
            "particles_down": None if particles_down is None else int(particles_down),
            "symmetries": symmetries,
            "reverse": True,
        }


__all__ = [
    "boson_basis_1d",
    "boson_basis_general",
    "ho_basis",
    "spin_basis_1d",
    "spin_basis_general",
    "spinful_fermion_basis_1d",
    "spinful_fermion_basis_general",
    "spinless_fermion_basis_1d",
    "spinless_fermion_basis_general",
]
