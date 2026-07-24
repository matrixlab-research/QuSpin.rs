use approx::assert_abs_diff_eq;
use qmbed::Complex64;
use qmbed::workflow::track_states;

#[test]
fn state_tracking_is_permutation_and_phase_invariant() {
    let previous = vec![
        vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
    ];
    let current = vec![
        vec![Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0)],
        vec![Complex64::new(-1.0, 0.0), Complex64::new(0.0, 0.0)],
    ];
    let tracked = track_states(&previous, &current, 1.0e-8).unwrap();
    assert_eq!(tracked.permutation, vec![1, 0]);
    assert!(tracked.ambiguous.is_empty());
    assert_eq!(tracked.overlaps, vec![1.0, 1.0]);
    assert_abs_diff_eq!(tracked.phases[0].re, -1.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(tracked.phases[1].im, 1.0, epsilon = 1.0e-12);
}

#[test]
fn state_tracking_reports_ambiguous_frames() {
    let amplitude = 1.0 / 2.0_f64.sqrt();
    let previous = vec![
        vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
    ];
    let current = vec![
        vec![
            Complex64::new(amplitude, 0.0),
            Complex64::new(amplitude, 0.0),
        ],
        vec![
            Complex64::new(amplitude, 0.0),
            Complex64::new(-amplitude, 0.0),
        ],
    ];
    let tracked = track_states(&previous, &current, 1.0e-12).unwrap();
    assert_eq!(tracked.ambiguous, vec![0, 1]);
}
