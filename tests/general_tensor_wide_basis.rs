use approx::assert_abs_diff_eq;
use quspin::Complex64;
use quspin::basis::{
    Basis, BasisProjector, BosonBasis1D, ClosureSymmetryMap, GeneralBasis, PhotonBasis,
    SpinBasis1D, SymmetrySector, TensorBasis, U256, UserBasis,
};
use quspin::measure::project_operator;
use quspin::operator::{Coupling, LinearOperator, MatrixFormat, OperatorBuilder, OperatorTerm};

fn periodic_heisenberg(sites: usize) -> Vec<OperatorTerm> {
    let mut zz = Vec::new();
    let mut plus_minus = Vec::new();
    let mut minus_plus = Vec::new();
    for site in 0..sites {
        let next = (site + 1) % sites;
        zz.push(Coupling::new(1.0, vec![site, next]));
        plus_minus.push(Coupling::new(0.5, vec![site, next]));
        minus_plus.push(Coupling::new(0.5, vec![site, next]));
    }
    vec![
        OperatorTerm::new("zz", zz).unwrap(),
        OperatorTerm::new("+-", plus_minus).unwrap(),
        OperatorTerm::new("-+", minus_plus).unwrap(),
    ]
}

#[test]
fn closure_symmetry_map_reproduces_the_builtin_translation_sector() {
    let parent = SpinBasis1D::builder(4).up(2).build().unwrap();
    let translation = ClosureSymmetryMap::new(4, |state: u128| {
        Ok((
            ((state << 1) & 0b1111) | (state >> 3),
            Complex64::new(1.0, 0.0),
        ))
    })
    .unwrap();
    let general =
        GeneralBasis::new(parent, SymmetrySector::new().with_map(translation, 1)).unwrap();
    let builtin = SpinBasis1D::builder(4).up(2).momentum(1).build().unwrap();
    assert_eq!(general.len(), builtin.len());
    for index in 0..general.len() {
        assert_eq!(general.state(index).unwrap(), builtin.state(index).unwrap());
    }

    let terms = periodic_heisenberg(4);
    let general_operator = OperatorBuilder::on(&general)
        .terms(terms.clone())
        .build(MatrixFormat::Dense)
        .unwrap();
    let full_operator = OperatorBuilder::on(general.parent())
        .terms(periodic_heisenberg(4))
        .build(MatrixFormat::Csc)
        .unwrap();
    let projector = BasisProjector::from_general(&general).unwrap();
    let projected = project_operator(&full_operator, &projector, MatrixFormat::Dense).unwrap();
    let builtin_operator = OperatorBuilder::on(&builtin)
        .terms(terms)
        .build(MatrixFormat::Dense)
        .unwrap();
    for (actual, expected) in general_operator
        .to_dense()
        .iter()
        .zip(builtin_operator.to_dense())
    {
        assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-12);
        assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-12);
    }
    for (actual, expected) in projected.to_dense().iter().zip(general_operator.to_dense()) {
        assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-12);
        assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-12);
    }

    let reduced: Vec<_> = (0..projector.reduced_dimension())
        .map(|index| Complex64::new(0.3 * (index + 1) as f64, -0.2))
        .collect();
    let mut lifted = vec![Complex64::new(0.0, 0.0); projector.source_dimension()];
    projector.apply(&reduced, &mut lifted).unwrap();
    let mut recovered = vec![Complex64::new(0.0, 0.0); projector.reduced_dimension()];
    projector.project(&lifted, &mut recovered).unwrap();
    for (actual, expected) in recovered.iter().zip(reduced) {
        assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-12);
        assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-12);
    }
}

#[test]
fn tensor_basis_applies_each_factor_without_kronecker_materialization() {
    let spin = SpinBasis1D::builder(1).pauli(true).build().unwrap();
    let boson = BosonBasis1D::builder(1, 2).build().unwrap();
    let tensor = TensorBasis::new(spin, boson).unwrap();
    assert_eq!(tensor.len(), 4);
    let number_weighted_spin = OperatorBuilder::on(&tensor)
        .term(OperatorTerm::new("z|n", [Coupling::new(1.0, vec![0, 0])]).unwrap())
        .build(MatrixFormat::Csc)
        .unwrap();
    let diagonal = number_weighted_spin.diagonal();
    assert_eq!(diagonal.len(), 4);
    assert_abs_diff_eq!(diagonal[0].re, 0.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(diagonal[1].re, -1.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(diagonal[2].re, 0.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(diagonal[3].re, 1.0, epsilon = 1.0e-12);
}

#[test]
fn photon_basis_enforces_total_excitation_and_exchange_dynamics() {
    let matter = SpinBasis1D::builder(1).build().unwrap();
    let photon = BosonBasis1D::builder(1, 3).build().unwrap();
    let basis = PhotonBasis::fixed_total_excitations(matter, photon, 1, |state| {
        state.count_ones() as usize
    })
    .unwrap();
    assert_eq!(basis.len(), 2);
    assert_eq!(basis.total_excitations(), Some(1));
    let exchange = OperatorBuilder::on(&basis)
        .terms([
            OperatorTerm::new("+|-", [Coupling::new(1.0, vec![0, 0])]).unwrap(),
            OperatorTerm::new("-|+", [Coupling::new(1.0, vec![0, 0])]).unwrap(),
        ])
        .build(MatrixFormat::Csc)
        .unwrap();
    assert_eq!(exchange.nnz(), 2);
    let dense = exchange.to_dense();
    assert_abs_diff_eq!(dense[1].re, 1.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(dense[2].re, 1.0, epsilon = 1.0e-12);
}

#[test]
fn wide_user_basis_actions_reach_beyond_u128() {
    let vacuum = U256::zero();
    let occupied = vacuum.with_bit(200, true).unwrap();
    assert!(occupied.bit(200).unwrap());
    assert_eq!(occupied.count_ones(), 1);
    let basis = UserBasis::builder(256)
        .states([vacuum, occupied])
        .operator('n', |state: U256, site| {
            Ok(state
                .bit(site)?
                .then_some((state, Complex64::new(1.0, 0.0))))
        })
        .build()
        .unwrap();
    let number = OperatorBuilder::on(&basis)
        .term(OperatorTerm::new("n", [Coupling::new(1.0, vec![200])]).unwrap())
        .build(MatrixFormat::Csc)
        .unwrap();
    assert_eq!(number.diagonal()[0], Complex64::new(0.0, 0.0));
    assert_eq!(number.diagonal()[1], Complex64::new(1.0, 0.0));
}
