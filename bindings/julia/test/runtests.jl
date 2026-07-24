using Test
using QMBED

const coupling = Coupling(1.0, [0, 1])
const half_coupling = Coupling(0.5, [0, 1])
const terms = OperatorSpec[
    OperatorSpec(OpProduct([ZOp, ZOp]), [coupling]),
    QMBED.Compat.QuSpin.operator_term("+-", [half_coupling]),
    QMBED.Compat.QuSpin.operator_term("-+", [half_coupling]),
]
const result = eigsh(SpinBasis(sites=2), terms, EigshOptions(eigenpairs=2))

@test result.dimension == 4
@test result.eigenvalues[1] ≈ -0.75 atol=1.0e-10
@test result.converged
