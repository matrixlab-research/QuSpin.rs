from __future__ import annotations

from collections.abc import Iterable, Sequence

from ..model import (
    Coupling,
    EigshOptions,
    LocalOperator,
    OpProduct,
    OperatorSpec,
    eigsh as native_eigsh,
)

TARGET_QUSPIN_VERSION = "1.0.1"

_SYMBOLS = {
    "I": LocalOperator.IDENTITY,
    "n": LocalOperator.NUMBER,
    "z": LocalOperator.Z,
    "+": LocalOperator.RAISING,
    "-": LocalOperator.LOWERING,
    "x": LocalOperator.X,
    "y": LocalOperator.Y,
}


def operator_term(
    operator: str,
    couplings: Iterable[Coupling],
) -> OperatorSpec:
    if operator.count("|") > 1:
        raise ValueError("a spinful operator may contain only one separator")
    split = operator.find("|")
    split = None if split < 0 else split
    local = tuple(
        _SYMBOLS.get(symbol, f"custom:{symbol}")
        for symbol in operator
        if symbol != "|"
    )
    return OperatorSpec(OpProduct(local, split), tuple(couplings))


def terms_from_static(static: Iterable[Sequence]) -> tuple[OperatorSpec, ...]:
    terms = []
    for operator, coupling_rows in static:
        couplings = []
        for coefficient, *sites in coupling_rows:
            couplings.append(Coupling(complex(coefficient), tuple(sites)))
        terms.append(operator_term(operator, couplings))
    return tuple(terms)


def eigsh(
    basis,
    static: Iterable[Sequence],
    *,
    k: int,
    which: str = "SA",
    sigma: float | None = None,
    **options,
):
    targets = {
        "SA": "smallest_algebraic",
        "LA": "largest_algebraic",
        "SM": "smallest_magnitude",
        "LM": "largest_magnitude",
        "BE": "both_ends",
    }
    target = "shift" if sigma is not None else targets[which]
    return native_eigsh(
        basis,
        terms_from_static(static),
        EigshOptions(k, target=target, shift=sigma, **options),
    )
