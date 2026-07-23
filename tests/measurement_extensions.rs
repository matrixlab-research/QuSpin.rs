use approx::assert_abs_diff_eq;
use quspin::Complex64;
use quspin::measure::{
    EntropyOrder, array_to_ints, array_to_states, density_expectation, diagonal_ensemble,
    diagonal_ensemble_density, diagonal_ensemble_observable, ed_density_vs_time, ed_state_vs_time,
    energy_window_indices, entanglement_entropy, entanglement_entropy_batch,
    entanglement_entropy_density, entanglement_entropy_density_subsystem,
    entanglement_entropy_subsystem, entanglement_spectrum, entanglement_spectrum_density,
    entanglement_spectrum_subsystem, expectation, ints_to_array, kl_divergence, matrix_element,
    mean_level_spacing, observables_vs_time, partial_trace, partial_trace_density,
    partial_trace_density_subsystem, partial_trace_subsystem, quantum_fluctuation, states_to_array,
};
use quspin::operator::Operator;
use quspin::solve::StateTrajectory;

fn sigma_z() -> Operator {
    Operator::from_dense(
        2,
        2,
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-1.0, 0.0),
        ],
    )
    .unwrap()
}

#[test]
fn observables_and_fluctuations_match_two_level_anchors() {
    let operator = sigma_z();
    let plus = vec![
        Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
    ];
    assert_abs_diff_eq!(
        expectation(&operator, &plus).unwrap().re,
        0.0,
        epsilon = 1.0e-12
    );
    assert_abs_diff_eq!(
        matrix_element(&plus, &operator, &plus).unwrap().re,
        0.0,
        epsilon = 1.0e-12
    );
    assert_abs_diff_eq!(
        quantum_fluctuation(&operator, &plus).unwrap(),
        1.0,
        epsilon = 1.0e-12
    );

    let trajectory = StateTrajectory {
        times: vec![0.0, 1.0],
        states: vec![
            plus.clone(),
            vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        ],
    };
    let values = observables_vs_time(&trajectory, &[("z".to_string(), &operator)]).unwrap();
    assert_abs_diff_eq!(values["z"][0].re, 0.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(values["z"][1].re, 1.0, epsilon = 1.0e-12);
}

#[test]
fn partial_trace_and_entropy_distinguish_product_and_bell_states() {
    let product = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
    ];
    assert_abs_diff_eq!(
        entanglement_entropy(&product, 2, 2, EntropyOrder::VonNeumann).unwrap(),
        0.0,
        epsilon = 1.0e-12
    );

    let amplitude = 1.0 / 2.0_f64.sqrt();
    let bell = vec![
        Complex64::new(amplitude, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(amplitude, 0.0),
    ];
    let reduced = partial_trace(&bell, 2, 2).unwrap();
    assert_abs_diff_eq!(reduced[0].re, 0.5, epsilon = 1.0e-12);
    assert_abs_diff_eq!(reduced[3].re, 0.5, epsilon = 1.0e-12);
    assert_abs_diff_eq!(
        entanglement_entropy(&bell, 2, 2, EntropyOrder::VonNeumann).unwrap(),
        2.0_f64.ln(),
        epsilon = 1.0e-12
    );
    assert_abs_diff_eq!(
        entanglement_entropy(&bell, 2, 2, EntropyOrder::Renyi(2.0)).unwrap(),
        2.0_f64.ln(),
        epsilon = 1.0e-12
    );
}

#[test]
fn ensemble_statistics_and_state_conversions_are_deterministic() {
    let eigenvectors = vec![
        vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
    ];
    let initial = vec![
        Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
    ];
    let ensemble = diagonal_ensemble(&[-1.0, 1.0], &eigenvectors, &initial).unwrap();
    assert_abs_diff_eq!(ensemble.mean_energy, 0.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(ensemble.energy_variance, 1.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(ensemble.entropy, 2.0_f64.ln(), epsilon = 1.0e-12);
    assert_abs_diff_eq!(
        kl_divergence(&[0.5, 0.5], &[0.5, 0.5]).unwrap(),
        0.0,
        epsilon = 1.0e-12
    );
    assert_abs_diff_eq!(
        mean_level_spacing(&[0.0, 1.0, 3.0, 6.0]).unwrap(),
        (0.5 + 2.0 / 3.0) / 2.0,
        epsilon = 1.0e-12
    );

    let states = vec![0_u128, 5, 15];
    let occupations = states_to_array(&states, 4, 2).unwrap();
    assert_eq!(array_to_states(&occupations, 2).unwrap(), states);
    let binary = ints_to_array(&states, 4).unwrap();
    assert_eq!(binary[1], vec![0, 1, 0, 1]);
    assert_eq!(array_to_ints(&binary).unwrap(), states);
}

#[test]
fn exact_eigenbasis_evolution_supports_pure_and_mixed_states() {
    let eigenvalues = [-1.0, 1.0];
    let eigenvectors = vec![
        vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
    ];
    let initial = vec![
        Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
    ];
    let trajectory = ed_state_vs_time(&initial, &eigenvalues, &eigenvectors, &[0.0, 0.5]).unwrap();
    assert_abs_diff_eq!(trajectory.states[1][0].arg(), 0.5, epsilon = 1.0e-12);
    assert_abs_diff_eq!(trajectory.states[1][1].arg(), -0.5, epsilon = 1.0e-12);

    let density = vec![
        Complex64::new(0.5, 0.0),
        Complex64::new(0.5, 0.0),
        Complex64::new(0.5, 0.0),
        Complex64::new(0.5, 0.0),
    ];
    let evolved = ed_density_vs_time(&density, &eigenvalues, &eigenvectors, &[0.5]).unwrap();
    assert_abs_diff_eq!(evolved[0][0].re + evolved[0][3].re, 1.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(evolved[0][1].norm(), 0.5, epsilon = 1.0e-12);
}

#[test]
fn mixed_and_batched_measurements_share_the_pure_state_limits() {
    let amplitude = 1.0 / 2.0_f64.sqrt();
    let bell = vec![
        Complex64::new(amplitude, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(amplitude, 0.0),
    ];
    let density: Vec<_> = bell
        .iter()
        .flat_map(|left| bell.iter().map(move |right| *left * right.conj()))
        .collect();
    assert_eq!(
        partial_trace_density(&density, 2, 2).unwrap(),
        partial_trace(&bell, 2, 2).unwrap()
    );
    let pure_spectrum = entanglement_spectrum(&bell, 2, 2).unwrap();
    let mixed_spectrum = entanglement_spectrum_density(&density, 2, 2).unwrap();
    for (actual, expected) in mixed_spectrum.iter().zip(pure_spectrum) {
        assert_abs_diff_eq!(actual, &expected, epsilon = 1.0e-12);
    }
    assert_abs_diff_eq!(
        entanglement_entropy_density(&density, 2, 2, EntropyOrder::VonNeumann).unwrap(),
        2.0_f64.ln(),
        epsilon = 1.0e-12
    );
    let product = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
    ];
    let batch =
        entanglement_entropy_batch(&[bell, product], 2, 2, EntropyOrder::VonNeumann).unwrap();
    assert_abs_diff_eq!(batch[0], 2.0_f64.ln(), epsilon = 1.0e-12);
    assert_abs_diff_eq!(batch[1], 0.0, epsilon = 1.0e-12);
}

#[test]
fn density_diagonal_ensemble_and_observable_are_consistent() {
    let eigenvectors = vec![
        vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
    ];
    let density = vec![
        Complex64::new(0.75, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.25, 0.0),
    ];
    let ensemble = diagonal_ensemble_density(&[-1.0, 1.0], &eigenvectors, &density).unwrap();
    assert_eq!(ensemble.probabilities, vec![0.75, 0.25]);
    let z = sigma_z();
    assert_abs_diff_eq!(
        density_expectation(&z, &density).unwrap().re,
        0.5,
        epsilon = 1.0e-12
    );
    assert_abs_diff_eq!(
        diagonal_ensemble_observable(&ensemble, &eigenvectors, &z)
            .unwrap()
            .re,
        0.5,
        epsilon = 1.0e-12
    );
    assert_eq!(
        energy_window_indices(&[-2.0, -0.1, 0.2, 3.0], 0.0, 0.25).unwrap(),
        vec![1, 2]
    );
}

#[test]
fn arbitrary_site_partial_trace_supports_noncontiguous_subsystems() {
    let amplitude = 1.0 / 2.0_f64.sqrt();
    let mut ghz = vec![Complex64::new(0.0, 0.0); 8];
    ghz[0] = Complex64::new(amplitude, 0.0);
    ghz[7] = Complex64::new(amplitude, 0.0);
    let reduced = partial_trace_subsystem(&ghz, &[2, 2, 2], &[0, 2]).unwrap();
    assert_eq!(reduced.len(), 16);
    assert_abs_diff_eq!(reduced[0].re, 0.5, epsilon = 1.0e-12);
    assert_abs_diff_eq!(reduced[15].re, 0.5, epsilon = 1.0e-12);
    assert_abs_diff_eq!(reduced[3].norm(), 0.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(
        entanglement_entropy_subsystem(&ghz, &[2, 2, 2], &[0, 2], EntropyOrder::VonNeumann)
            .unwrap(),
        2.0_f64.ln(),
        epsilon = 1.0e-12
    );
    let spectrum = entanglement_spectrum_subsystem(&ghz, &[2, 2, 2], &[0, 2]).unwrap();
    assert_abs_diff_eq!(spectrum[2], 0.5, epsilon = 1.0e-12);
    assert_abs_diff_eq!(spectrum[3], 0.5, epsilon = 1.0e-12);

    let density: Vec<_> = ghz
        .iter()
        .flat_map(|left| ghz.iter().map(move |right| *left * right.conj()))
        .collect();
    let mixed = partial_trace_density_subsystem(&density, &[2, 2, 2], &[0, 2]).unwrap();
    for (actual, expected) in mixed.iter().zip(reduced) {
        assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-12);
        assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-12);
    }
    assert_abs_diff_eq!(
        entanglement_entropy_density_subsystem(
            &density,
            &[2, 2, 2],
            &[0, 2],
            EntropyOrder::Renyi(2.0),
        )
        .unwrap(),
        2.0_f64.ln(),
        epsilon = 1.0e-12
    );
}
