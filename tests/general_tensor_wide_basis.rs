use approx::assert_abs_diff_eq;
use num_bigint::BigUint;
use qmbed::Complex64;
use qmbed::basis::{
    Basis, BasisProjector, BosonBasis1D, ClosureSymmetryMap, GeneralBasis, PhotonBasis,
    SpinBasis1D, SpinfulFermionBasis1D, StateStorage, SymmetrySector, TensorBasis, U256, UserBasis,
    WideSpinBasis256, basis_int_to_python_int, basis_ones, basis_zeros, bitwise_and,
    bitwise_leftshift, bitwise_not, bitwise_or, bitwise_rightshift, bitwise_xor, coherent_state,
    get_basis_type, photon_hspace_dim, python_int_to_basis_int, state_from_biguint,
    state_to_biguint,
};
use qmbed::measure::project_operator;
use qmbed::operator::{
    Coupling, LinearOperator, MatrixFormat, OperatorBuilder, OperatorTerm, apply_sector_shift,
    bra_ket_transitions,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

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
    assert!(
        projector
            .preserves_operator_symmetry(&full_operator, 1.0e-12)
            .unwrap()
    );
    let local_z = OperatorBuilder::on(general.parent())
        .term(OperatorTerm::new("z", [Coupling::new(1.0, vec![0])]).unwrap())
        .build(MatrixFormat::Csc)
        .unwrap();
    assert!(projector.symmetry_leakage_norm(&local_z).unwrap() > 1.0e-6);
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
fn cross_sector_builder_reduces_into_the_target_symmetry_sector() {
    let translation = || {
        ClosureSymmetryMap::new(4, |state: u128| {
            Ok((
                ((state << 1) & 0b1111) | (state >> 3),
                Complex64::new(1.0, 0.0),
            ))
        })
        .unwrap()
    };
    let source = GeneralBasis::new(
        SpinBasis1D::builder(4).up(0).build().unwrap(),
        SymmetrySector::new().with_map(translation(), 0),
    )
    .unwrap();
    let target = GeneralBasis::new(
        SpinBasis1D::builder(4).up(1).build().unwrap(),
        SymmetrySector::new().with_map(translation(), 1),
    )
    .unwrap();
    let couplings = (0..4)
        .map(|site| {
            Coupling::new(
                Complex64::from_polar(1.0, -std::f64::consts::TAU * site as f64 / 4.0),
                vec![site],
            )
        })
        .collect::<Vec<_>>();
    let term = OperatorTerm::new("+", couplings).unwrap();
    let reduced = OperatorBuilder::between(&source, &target)
        .term(term.clone())
        .build(MatrixFormat::Csc)
        .unwrap();
    let mut streamed = vec![Complex64::new(0.0, 0.0); target.len()];
    apply_sector_shift(
        &source,
        &target,
        std::slice::from_ref(&term),
        &[Complex64::new(1.0, 0.0)],
        &mut streamed,
    )
    .unwrap();
    assert_eq!(streamed, reduced.to_dense());
    let full = OperatorBuilder::between(source.parent(), target.parent())
        .term(term)
        .build(MatrixFormat::Csc)
        .unwrap();
    let source_projector = BasisProjector::from_general(&source).unwrap();
    let target_projector = BasisProjector::from_general(&target).unwrap();
    let mut full_source = vec![Complex64::new(0.0, 0.0); source_projector.source_dimension()];
    source_projector
        .apply(&[Complex64::new(1.0, 0.0)], &mut full_source)
        .unwrap();
    let mut full_target = vec![Complex64::new(0.0, 0.0); target_projector.source_dimension()];
    full.apply(&full_source, &mut full_target).unwrap();
    let mut expected = vec![Complex64::new(0.0, 0.0); target_projector.reduced_dimension()];
    target_projector
        .project(&full_target, &mut expected)
        .unwrap();
    let actual = reduced.to_dense()[0];
    assert!(actual.norm() > 1.0);
    assert_abs_diff_eq!(actual.re, expected[0].re, epsilon = 1.0e-12);
    assert_abs_diff_eq!(actual.im, expected[0].im, epsilon = 1.0e-12);
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

#[test]
fn wide_integer_helpers_and_photon_utilities_cover_python_helper_semantics() {
    let low: U256 = python_int_to_basis_int(3);
    let high = bitwise_leftshift(low, 130);
    assert!(high.bit(130).unwrap());
    assert!(high.bit(131).unwrap());
    assert_eq!(bitwise_rightshift(high, 130), low);
    assert_eq!(
        bitwise_and(low, python_int_to_basis_int(1)),
        python_int_to_basis_int(1)
    );
    assert_eq!(
        bitwise_or(low, python_int_to_basis_int(4)),
        python_int_to_basis_int(7)
    );
    assert_eq!(
        bitwise_xor(low, python_int_to_basis_int(1)),
        python_int_to_basis_int(2)
    );
    assert_eq!(bitwise_not(bitwise_not(low)), low);
    assert_eq!(basis_int_to_python_int(low).unwrap(), 3);
    assert_eq!(basis_zeros::<4>(2), vec![U256::zero(); 2]);
    assert_eq!(basis_ones::<4>(1)[0].count_ones(), 256);
    assert_eq!(get_basis_type(200, None, 2).unwrap(), StateStorage::U256);
    let arbitrary = (BigUint::from(1_u8) << 200) + BigUint::from(7_u8);
    let encoded: U256 = state_from_biguint(&arbitrary).unwrap();
    assert_eq!(state_to_biguint(encoded), arbitrary);

    let coherent = coherent_state(Complex64::new(0.0, 0.0), 4).unwrap();
    assert_eq!(coherent[0], Complex64::new(1.0, 0.0));
    assert!(coherent[1..].iter().all(|value| value.norm() == 0.0));
    assert_eq!(photon_hspace_dim(2, Some(1), Some(2)).unwrap(), 3);
    assert_eq!(photon_hspace_dim(4, None, Some(3)).unwrap(), 64);
    assert_eq!(photon_hspace_dim(8, Some(4), None).unwrap(), 163);
}

#[test]
fn user_basis_can_defer_state_enumeration_until_materialization() {
    let calls = Arc::new(AtomicUsize::new(0));
    let factory_calls = calls.clone();
    let builder = UserBasis::builder(3)
        .deferred_states(move || {
            factory_calls.fetch_add(1, Ordering::SeqCst);
            Ok(vec![0_u128, 1, 2])
        })
        .operator('I', |state, _| Ok(Some((state, Complex64::new(1.0, 0.0)))));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let basis = builder.materialize().unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(basis.len(), 3);
}

#[test]
fn spinful_sector_unions_and_majorana_operators_obey_clifford_algebra() {
    let union = SpinfulFermionBasis1D::builder(2)
        .particle_sectors([(1, 0), (0, 1)])
        .build()
        .unwrap();
    assert_eq!(union.len(), 4);

    let basis = SpinfulFermionBasis1D::builder(1).build().unwrap();
    let x = OperatorBuilder::on(&basis)
        .term(OperatorTerm::new("x|", [Coupling::new(1.0, vec![0])]).unwrap())
        .build(MatrixFormat::Dense)
        .unwrap();
    let y = OperatorBuilder::on(&basis)
        .term(OperatorTerm::new("y|", [Coupling::new(1.0, vec![0])]).unwrap())
        .build(MatrixFormat::Dense)
        .unwrap();
    assert_eq!(x.adjoint().unwrap().to_dense(), x.to_dense());
    assert_eq!(y.adjoint().unwrap().to_dense(), y.to_dense());
    assert_eq!(
        x.pow(2).unwrap().diagonal(),
        vec![Complex64::new(1.0, 0.0); 4]
    );
    assert_eq!(
        y.pow(2).unwrap().diagonal(),
        vec![Complex64::new(1.0, 0.0); 4]
    );
    assert_eq!(
        x.product(&y)
            .unwrap()
            .add(&y.product(&x).unwrap())
            .unwrap()
            .nnz(),
        0
    );
}

#[test]
fn branching_and_parallel_user_basis_paths_preserve_transition_semantics() {
    let serial = UserBasis::builder(9)
        .state_filter(|state| state.count_ones() == 2 && state & (state << 1) == 0)
        .unwrap()
        .operator('n', |state, site| {
            Ok(((state >> site) & 1 == 1).then_some((state, Complex64::new(1.0, 0.0))))
        })
        .build()
        .unwrap();
    let parallel = UserBasis::builder(9)
        .state_filter_parallel(|state| state.count_ones() == 2 && state & (state << 1) == 0)
        .unwrap()
        .operator('n', |state, site| {
            Ok(((state >> site) & 1 == 1).then_some((state, Complex64::new(1.0, 0.0))))
        })
        .build()
        .unwrap();
    assert_eq!(serial.len(), parallel.len());
    for index in 0..serial.len() {
        assert_eq!(serial.state(index).unwrap(), parallel.state(index).unwrap());
    }

    let qutrit = UserBasis::builder(1)
        .states([0_u128, 1, 2])
        .branching_operator('a', |state, _| {
            Ok((0_u128..3)
                .map(|target| (target, Complex64::new((target + 1) as f64, state as f64)))
                .collect())
        })
        .build()
        .unwrap();
    let matrix = OperatorBuilder::on(&qutrit)
        .term(OperatorTerm::new("a", [Coupling::new(1.0, vec![0])]).unwrap())
        .checks(qmbed::operator::AssemblyChecks {
            hermiticity: false,
            particle_conservation: false,
            symmetry_compatibility: true,
        })
        .build(MatrixFormat::Csc)
        .unwrap()
        .to_dense();
    for row in 0..3 {
        for column in 0..3 {
            assert_eq!(
                matrix[row * 3 + column],
                Complex64::new((row + 1) as f64, column as f64)
            );
        }
    }
}

#[test]
fn wide_spin_basis_assembles_high_site_actions_without_u128_conversion() {
    let basis = WideSpinBasis256::new(201, Some(1), false).unwrap();
    assert_eq!(basis.len(), 201);
    let high = U256::zero().with_bit(200, true).unwrap();
    let source = basis.index(high).unwrap();
    let lowering =
        OperatorBuilder::between(&basis, &WideSpinBasis256::new(201, Some(0), false).unwrap())
            .term(OperatorTerm::new("-", [Coupling::new(1.0, vec![200])]).unwrap())
            .build(MatrixFormat::Csc)
            .unwrap();
    assert_eq!(lowering.shape(), (1, 201));
    assert_eq!(lowering.to_dense()[source], Complex64::new(1.0, 0.0));
    let transitions = bra_ket_transitions(&basis, "-", &[200], 1.0, [high]).unwrap();
    assert_eq!(transitions.len(), 1);
    assert_eq!(transitions[0].ket, high);
    assert_eq!(transitions[0].bra, U256::zero());
    assert_eq!(transitions[0].matrix_element, Complex64::new(1.0, 0.0));
}
