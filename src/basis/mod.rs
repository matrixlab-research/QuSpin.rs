use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::sync::Arc;

use num_complex::Complex64;

use crate::operator::{LinearOperator, MatrixFormat, check_apply_shape};
use crate::{QuSpinError, Result};

/// Finite Hilbert-space basis and its local operator semantics.
pub trait Basis: Send + Sync {
    type State: Copy + Eq + Send + Sync;

    fn len(&self) -> usize;
    fn state(&self, index: usize) -> Result<Self::State>;
    fn index(&self, state: Self::State) -> Result<usize>;
    fn apply_local(
        &self,
        state: Self::State,
        operator: &str,
        sites: &[usize],
    ) -> Result<Option<(Self::State, Complex64)>>;

    /// Applies a local operator and returns every nonzero destination.
    ///
    /// Most spin-one-half, boson ladder, and fermion strings are deterministic,
    /// so the default implementation wraps [`Basis::apply_local`]. Higher-spin
    /// Cartesian operators and general user-defined local matrices can branch
    /// and override this method without changing the universal assembler.
    fn apply_local_transitions(
        &self,
        state: Self::State,
        operator: &str,
        sites: &[usize],
    ) -> Result<Vec<(Self::State, Complex64)>> {
        Ok(self
            .apply_local(state, operator, sites)?
            .into_iter()
            .collect())
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

fn fixed_weight_states(sites: usize, particles: Option<usize>) -> Result<Vec<u128>> {
    if sites > 128 {
        return Err(QuSpinError::UnsupportedBackend(
            "the initial u128 state backend supports at most 128 orbitals".into(),
        ));
    }
    if particles.is_some_and(|count| count > sites) {
        return Err(QuSpinError::InvalidSector(
            "particle count exceeds site count".into(),
        ));
    }
    let limit = 1_u128
        .checked_shl(u32::try_from(sites).unwrap_or(u32::MAX))
        .ok_or_else(|| QuSpinError::UnsupportedBackend("state enumeration overflow".into()))?;
    let mut states = Vec::new();
    for state in 0..limit {
        if particles.is_none_or(|count| state.count_ones() as usize == count) {
            states.push(state);
        }
    }
    Ok(states)
}

fn fixed_digit_sum_states(
    sites: usize,
    states_per_site: usize,
    total: Option<usize>,
) -> Result<Vec<u128>> {
    if sites == 0 || states_per_site == 0 {
        return Err(QuSpinError::InvalidSector(
            "sites and local state count must be positive".into(),
        ));
    }
    if total.is_some_and(|value| value > sites.saturating_mul(states_per_site - 1)) {
        return Err(QuSpinError::InvalidSector(
            "requested occupation exceeds the local spin capacity".into(),
        ));
    }
    let base = states_per_site as u128;
    let exponent = u32::try_from(sites)
        .map_err(|_| QuSpinError::UnsupportedBackend("site count is too large".into()))?;
    let limit = base.checked_pow(exponent).ok_or_else(|| {
        QuSpinError::UnsupportedBackend("mixed-radix state encoding overflow".into())
    })?;
    if total.is_none() {
        return Ok((0..limit).collect());
    }

    fn enumerate(
        site: usize,
        sites: usize,
        states_per_site: usize,
        remaining: usize,
        place: u128,
        encoded: u128,
        output: &mut Vec<u128>,
    ) {
        if site == sites {
            if remaining == 0 {
                output.push(encoded);
            }
            return;
        }
        let remaining_sites = sites - site - 1;
        let maximum_tail = remaining_sites.saturating_mul(states_per_site - 1);
        for digit in 0..states_per_site {
            if digit > remaining || remaining - digit > maximum_tail {
                continue;
            }
            enumerate(
                site + 1,
                sites,
                states_per_site,
                remaining - digit,
                place * states_per_site as u128,
                encoded + digit as u128 * place,
                output,
            );
        }
    }

    let mut states = Vec::new();
    enumerate(
        0,
        sites,
        states_per_site,
        total.unwrap_or_default(),
        1,
        0,
        &mut states,
    );
    states.sort_unstable();
    Ok(states)
}

fn state_index(states: &[u128], state: u128) -> Result<usize> {
    states
        .binary_search(&state)
        .map_err(|_| QuSpinError::StateNotInBasis)
}

fn rotate_lattice_state(state: u128, shift: usize, sites: usize, base: u128) -> u128 {
    if sites == 0 {
        return state;
    }
    let shift = shift % sites;
    if shift == 0 {
        return state;
    }
    if base == 2 {
        let mask = if sites == 128 {
            u128::MAX
        } else {
            (1_u128 << sites) - 1
        };
        return ((state << shift) & mask) | (state >> (sites - shift));
    }
    let mut translated = 0_u128;
    let mut source_place = 1_u128;
    for site in 0..sites {
        let digit = (state / source_place) % base;
        let target_site = (site + shift) % sites;
        translated += digit * base.pow(u32::try_from(target_site).unwrap_or(u32::MAX));
        source_place *= base;
    }
    translated
}

fn reflect_lattice_state(state: u128, sites: usize, base: u128) -> u128 {
    let mut reflected = 0_u128;
    let mut source_place = 1_u128;
    for site in 0..sites {
        let digit = (state / source_place) % base;
        let target_site = sites - site - 1;
        reflected += digit * base.pow(u32::try_from(target_site).unwrap_or(u32::MAX));
        source_place *= base;
    }
    reflected
}

#[derive(Clone, Copy, Debug)]
struct SymmetryImage {
    representative: u128,
    phase: Complex64,
    orbit_size: usize,
}

type SymmetrySectorData = (
    Vec<u128>,
    Vec<usize>,
    HashMap<u128, SymmetryImage>,
    Option<usize>,
    Option<i8>,
);

fn spin_symmetry_sector(
    parent_states: Vec<u128>,
    sites: usize,
    base: u128,
    momentum: Option<i32>,
    parity: Option<i8>,
) -> Result<SymmetrySectorData> {
    if sites == 0 {
        return Err(QuSpinError::InvalidSector(
            "symmetry sectors require at least one site".into(),
        ));
    }
    if parity.is_some_and(|value| value != -1 && value != 1) {
        return Err(QuSpinError::InvalidSector(
            "parity must be either -1 or +1".into(),
        ));
    }
    let sites_i64 = i64::try_from(sites)
        .map_err(|_| QuSpinError::UnsupportedBackend("site count is too large".into()))?;
    let normalized_momentum = momentum.map(|value| i64::from(value).rem_euclid(sites_i64) as usize);
    if parity.is_some() && normalized_momentum.is_some_and(|value| value != 0 && 2 * value != sites)
    {
        return Err(QuSpinError::IncompatibleSymmetry(
            "parity can share a one-dimensional sector with momentum only at k=0 or k=pi".into(),
        ));
    }

    if momentum.is_none() && parity.is_none() {
        let orbit_sizes = vec![1; parent_states.len()];
        let lookup = parent_states
            .iter()
            .copied()
            .map(|state| {
                (
                    state,
                    SymmetryImage {
                        representative: state,
                        phase: Complex64::new(1.0, 0.0),
                        orbit_size: 1,
                    },
                )
            })
            .collect();
        return Ok((parent_states, orbit_sizes, lookup, None, None));
    }

    let parent_lookup: HashSet<_> = parent_states.iter().copied().collect();
    let translations = if momentum.is_some() { sites } else { 1 };
    let mut visited = HashSet::with_capacity(parent_states.len());
    let mut sectors = Vec::<(u128, usize)>::new();
    let mut lookup = HashMap::with_capacity(parent_states.len());

    for seed in parent_states {
        if visited.contains(&seed) {
            continue;
        }
        let mut orbit = HashSet::new();
        for shift in 0..translations {
            let translated = rotate_lattice_state(seed, shift, sites, base);
            orbit.insert(translated);
            if parity.is_some() {
                orbit.insert(reflect_lattice_state(translated, sites, base));
            }
        }
        if orbit.iter().any(|state| !parent_lookup.contains(state)) {
            return Err(QuSpinError::IncompatibleSymmetry(
                "symmetry map leaves the selected magnetization sector".into(),
            ));
        }
        visited.extend(orbit.iter().copied());
        let representative = *orbit.iter().min().ok_or_else(|| {
            QuSpinError::InvalidSector("symmetry generated an empty orbit".into())
        })?;

        let mut coefficients = HashMap::<u128, Complex64>::new();
        for shift in 0..translations {
            let angle = normalized_momentum.map_or(0.0, |value| {
                -std::f64::consts::TAU * (value * shift) as f64 / sites as f64
            });
            let character = Complex64::from_polar(1.0, angle);
            let translated = rotate_lattice_state(representative, shift, sites, base);
            *coefficients
                .entry(translated)
                .or_insert(Complex64::new(0.0, 0.0)) += character;
            if let Some(parity_value) = parity {
                let reflected = reflect_lattice_state(translated, sites, base);
                *coefficients
                    .entry(reflected)
                    .or_insert(Complex64::new(0.0, 0.0)) += f64::from(parity_value) * character;
            }
        }
        coefficients.retain(|_, coefficient| coefficient.norm() > 1.0e-12);
        if coefficients.is_empty() {
            continue;
        }
        let representative_coefficient =
            coefficients
                .get(&representative)
                .copied()
                .ok_or(QuSpinError::IncompatibleSymmetry(
                    "symmetry projection removed its orbit representative".into(),
                ))?;
        let gauge = representative_coefficient / representative_coefficient.norm();
        let norm = coefficients
            .values()
            .map(Complex64::norm_sqr)
            .sum::<f64>()
            .sqrt();
        let orbit_size = coefficients.len();
        let expected_magnitude = 1.0 / (orbit_size as f64).sqrt();
        for (&state, coefficient) in &coefficients {
            let normalized = *coefficient / (gauge * norm);
            if (normalized.norm() - expected_magnitude).abs() > 1.0e-10 {
                return Err(QuSpinError::IncompatibleSymmetry(
                    "symmetry projection does not define a one-dimensional orbit sector".into(),
                ));
            }
            lookup.insert(
                state,
                SymmetryImage {
                    representative,
                    phase: normalized / expected_magnitude,
                    orbit_size,
                },
            );
        }
        sectors.push((representative, orbit_size));
    }

    sectors.sort_by_key(|(representative, _)| *representative);
    if sectors.is_empty() {
        return Err(QuSpinError::InvalidSector(
            "the requested symmetry sector is empty".into(),
        ));
    }
    let (states, orbit_sizes) = sectors.into_iter().unzip();
    Ok((states, orbit_sizes, lookup, normalized_momentum, parity))
}

fn translate_fermion_state(state: u128, shift: usize, sites: usize) -> (u128, f64) {
    let normalized = shift % sites;
    if normalized == 0 {
        return (state, 1.0);
    }
    let translated = rotate_lattice_state(state, normalized, sites, 2);
    let wrapped_mask = ((1_u128 << normalized) - 1) << (sites - normalized);
    let wrapped = (state & wrapped_mask).count_ones() as usize;
    let retained = state.count_ones() as usize - wrapped;
    let sign = if wrapped * retained % 2 == 0 {
        1.0
    } else {
        -1.0
    };
    (translated, sign)
}

type FermionTranslationSector = (
    Vec<u128>,
    Vec<usize>,
    HashMap<u128, SymmetryImage>,
    Option<usize>,
);

fn fermion_translation_sector(
    parent_states: Vec<u128>,
    sites: usize,
    momentum: Option<i32>,
) -> Result<FermionTranslationSector> {
    if momentum.is_none() {
        let orbit_sizes = vec![1; parent_states.len()];
        let lookup = parent_states
            .iter()
            .copied()
            .map(|state| {
                (
                    state,
                    SymmetryImage {
                        representative: state,
                        phase: Complex64::new(1.0, 0.0),
                        orbit_size: 1,
                    },
                )
            })
            .collect();
        return Ok((parent_states, orbit_sizes, lookup, None));
    }
    if sites == 0 {
        return Err(QuSpinError::InvalidSector(
            "translation sectors require at least one site".into(),
        ));
    }
    let sites_i64 = i64::try_from(sites)
        .map_err(|_| QuSpinError::UnsupportedBackend("site count is too large".into()))?;
    let normalized = i64::from(momentum.unwrap_or_default()).rem_euclid(sites_i64) as usize;
    let mut visited = HashSet::with_capacity(parent_states.len());
    let mut sectors = Vec::<(u128, usize)>::new();
    let mut lookup = HashMap::with_capacity(parent_states.len());

    for seed in parent_states {
        if visited.contains(&seed) {
            continue;
        }
        let orbit: HashSet<_> = (0..sites)
            .map(|shift| translate_fermion_state(seed, shift, sites).0)
            .collect();
        visited.extend(orbit.iter().copied());
        let representative = *orbit.iter().min().ok_or_else(|| {
            QuSpinError::InvalidSector("translation generated an empty fermion orbit".into())
        })?;
        let mut coefficients = HashMap::<u128, Complex64>::new();
        for shift in 0..sites {
            let (translated, sign) = translate_fermion_state(representative, shift, sites);
            let angle = -std::f64::consts::TAU * (normalized * shift) as f64 / sites as f64;
            *coefficients
                .entry(translated)
                .or_insert(Complex64::new(0.0, 0.0)) += sign * Complex64::from_polar(1.0, angle);
        }
        coefficients.retain(|_, coefficient| coefficient.norm() > 1.0e-12);
        if coefficients.is_empty() {
            continue;
        }
        let representative_coefficient =
            coefficients
                .get(&representative)
                .copied()
                .ok_or(QuSpinError::IncompatibleSymmetry(
                    "translation projection removed its fermion representative".into(),
                ))?;
        let gauge = representative_coefficient / representative_coefficient.norm();
        let norm = coefficients
            .values()
            .map(Complex64::norm_sqr)
            .sum::<f64>()
            .sqrt();
        let orbit_size = coefficients.len();
        let expected_magnitude = 1.0 / (orbit_size as f64).sqrt();
        for (&state, coefficient) in &coefficients {
            let projected = *coefficient / (gauge * norm);
            if (projected.norm() - expected_magnitude).abs() > 1.0e-10 {
                return Err(QuSpinError::IncompatibleSymmetry(
                    "fermion translation does not define a one-dimensional orbit sector".into(),
                ));
            }
            lookup.insert(
                state,
                SymmetryImage {
                    representative,
                    phase: projected / expected_magnitude,
                    orbit_size,
                },
            );
        }
        sectors.push((representative, orbit_size));
    }
    sectors.sort_by_key(|(representative, _)| *representative);
    if sectors.is_empty() {
        return Err(QuSpinError::InvalidSector(
            "the requested fermion momentum sector is empty".into(),
        ));
    }
    let (states, orbit_sizes) = sectors.into_iter().unzip();
    Ok((states, orbit_sizes, lookup, Some(normalized)))
}

fn checked_site(site: usize, sites: usize) -> Result<()> {
    if site >= sites {
        Err(QuSpinError::InvalidSite { site, sites })
    } else {
        Ok(())
    }
}

fn operator_chars(operator: &str, sites: &[usize]) -> Result<Vec<char>> {
    let chars: Vec<_> = operator
        .chars()
        .filter(|character| *character != '|')
        .collect();
    if chars.len() != sites.len() {
        return Err(QuSpinError::InvalidCoupling(format!(
            "operator arity {} does not match {} sites",
            chars.len(),
            sites.len()
        )));
    }
    Ok(chars)
}

/// Spin-chain basis for the full or fixed-magnetization spin space.
#[derive(Clone, Debug)]
pub struct SpinBasis1D {
    sites: usize,
    spin_twice: u16,
    up: Option<usize>,
    pauli: bool,
    momentum: Option<usize>,
    parity: Option<i8>,
    orbit_lengths: Vec<usize>,
    symmetry_lookup: HashMap<u128, SymmetryImage>,
    states: Vec<u128>,
}

impl SpinBasis1D {
    pub fn builder(sites: usize) -> SpinBasisBuilder {
        SpinBasisBuilder {
            sites,
            spin_twice: 1,
            up: None,
            momentum: None,
            parity: None,
            pauli: false,
        }
    }

    pub const fn sites(&self) -> usize {
        self.sites
    }

    pub const fn spin_twice(&self) -> u16 {
        self.spin_twice
    }

    pub const fn up(&self) -> Option<usize> {
        self.up
    }

    pub const fn pauli(&self) -> bool {
        self.pauli
    }

    pub const fn momentum(&self) -> Option<usize> {
        self.momentum
    }

    pub const fn parity(&self) -> Option<i8> {
        self.parity
    }
}

#[derive(Clone, Debug)]
pub struct SpinBasisBuilder {
    sites: usize,
    spin_twice: u16,
    up: Option<usize>,
    momentum: Option<i32>,
    parity: Option<i8>,
    pauli: bool,
}

impl SpinBasisBuilder {
    pub const fn spin_twice(mut self, spin_twice: u16) -> Self {
        self.spin_twice = spin_twice;
        self
    }

    pub const fn up(mut self, up: usize) -> Self {
        self.up = Some(up);
        self
    }

    pub const fn magnetization(mut self, up: usize) -> Self {
        self.up = Some(up);
        self
    }

    pub const fn momentum(mut self, momentum: i32) -> Self {
        self.momentum = Some(momentum);
        self
    }

    pub const fn parity(mut self, parity: i8) -> Self {
        self.parity = Some(parity);
        self
    }

    pub const fn pauli(mut self, pauli: bool) -> Self {
        self.pauli = pauli;
        self
    }

    pub fn build(self) -> Result<SpinBasis1D> {
        if self.spin_twice == 0 {
            return Err(QuSpinError::InvalidSector(
                "spin_twice must be positive".into(),
            ));
        }
        if self.pauli && self.spin_twice != 1 {
            return Err(QuSpinError::InvalidOptions(
                "the Pauli convention is defined only for spin one-half".into(),
            ));
        }
        let states_per_site = usize::from(self.spin_twice) + 1;
        let parent_states = if self.spin_twice == 1 {
            fixed_weight_states(self.sites, self.up)?
        } else {
            fixed_digit_sum_states(self.sites, states_per_site, self.up)?
        };
        let (states, orbit_lengths, symmetry_lookup, momentum, parity) = spin_symmetry_sector(
            parent_states,
            self.sites,
            states_per_site as u128,
            self.momentum,
            self.parity,
        )?;
        if states.is_empty() {
            return Err(QuSpinError::InvalidSector("empty spin sector".into()));
        }
        Ok(SpinBasis1D {
            sites: self.sites,
            spin_twice: self.spin_twice,
            up: self.up,
            pauli: self.pauli,
            momentum,
            parity,
            orbit_lengths,
            symmetry_lookup,
            states,
        })
    }
}

impl Basis for SpinBasis1D {
    type State = u128;

    fn len(&self) -> usize {
        self.states.len()
    }

    fn state(&self, index: usize) -> Result<Self::State> {
        self.states
            .get(index)
            .copied()
            .ok_or(QuSpinError::StateNotInBasis)
    }

    fn index(&self, state: Self::State) -> Result<usize> {
        state_index(&self.states, state)
    }

    fn apply_local(
        &self,
        state: Self::State,
        operator: &str,
        sites: &[usize],
    ) -> Result<Option<(Self::State, Complex64)>> {
        let transitions = self.apply_local_transitions(state, operator, sites)?;
        match transitions.as_slice() {
            [] => Ok(None),
            [transition] => Ok(Some(*transition)),
            _ => Err(QuSpinError::UnsupportedBackend(
                "this higher-spin local action branches; use apply_local_transitions".into(),
            )),
        }
    }

    fn apply_local_transitions(
        &self,
        state: Self::State,
        operator: &str,
        sites: &[usize],
    ) -> Result<Vec<(Self::State, Complex64)>> {
        let source_state = state;
        let chars = operator_chars(operator, sites)?;
        let states_per_site = u128::from(self.spin_twice) + 1;
        let spin = f64::from(self.spin_twice) * 0.5;
        let mut branches = vec![(state, Complex64::new(1.0, 0.0))];
        for (&site, op) in sites.iter().zip(chars).rev() {
            checked_site(site, self.sites)?;
            let place = states_per_site.pow(u32::try_from(site).unwrap_or(u32::MAX));
            let mut next = Vec::with_capacity(branches.len().saturating_mul(2));
            for (encoded, amplitude) in branches {
                let digit = (encoded / place) % states_per_site;
                let magnetic = digit as f64 - spin;
                let raise = || {
                    if digit >= u128::from(self.spin_twice) {
                        None
                    } else {
                        let factor = (spin * (spin + 1.0) - magnetic * (magnetic + 1.0)).sqrt();
                        Some((encoded + place, Complex64::new(factor, 0.0)))
                    }
                };
                let lower = || {
                    if digit == 0 {
                        None
                    } else {
                        let factor = (spin * (spin + 1.0) - magnetic * (magnetic - 1.0)).sqrt();
                        Some((encoded - place, Complex64::new(factor, 0.0)))
                    }
                };
                match op {
                    'I' => next.push((encoded, amplitude)),
                    'z' => {
                        let factor = if self.pauli { 2.0 * magnetic } else { magnetic };
                        if factor != 0.0 {
                            next.push((encoded, amplitude * factor));
                        }
                    }
                    '+' => {
                        if let Some((target, factor)) = raise() {
                            next.push((target, amplitude * factor));
                        }
                    }
                    '-' => {
                        if let Some((target, factor)) = lower() {
                            next.push((target, amplitude * factor));
                        }
                    }
                    'x' => {
                        let scale = if self.pauli { 1.0 } else { 0.5 };
                        if let Some((target, factor)) = raise() {
                            next.push((target, amplitude * scale * factor));
                        }
                        if let Some((target, factor)) = lower() {
                            next.push((target, amplitude * scale * factor));
                        }
                    }
                    'y' => {
                        let scale = if self.pauli { 1.0 } else { 0.5 };
                        if let Some((target, factor)) = raise() {
                            next.push((target, amplitude * Complex64::new(0.0, -scale) * factor));
                        }
                        if let Some((target, factor)) = lower() {
                            next.push((target, amplitude * Complex64::new(0.0, scale) * factor));
                        }
                    }
                    _ => return Err(QuSpinError::InvalidOperator(op.to_string())),
                }
            }
            branches = next;
            if branches.is_empty() {
                return Ok(Vec::new());
            }
        }
        if self.momentum.is_some() || self.parity.is_some() {
            let source_index = self.index(source_state)?;
            let source_orbit = self.orbit_lengths[source_index];
            let mut reduced = HashMap::<u128, Complex64>::new();
            for (encoded, mut amplitude) in branches {
                let Some(image) = self.symmetry_lookup.get(&encoded) else {
                    continue;
                };
                amplitude *=
                    (source_orbit as f64 / image.orbit_size as f64).sqrt() * image.phase.conj();
                *reduced
                    .entry(image.representative)
                    .or_insert(Complex64::new(0.0, 0.0)) += amplitude;
            }
            let mut transitions: Vec<_> = reduced
                .into_iter()
                .filter(|(_, amplitude)| amplitude.norm() > f64::EPSILON)
                .collect();
            transitions.sort_by_key(|(encoded, _)| *encoded);
            return Ok(transitions);
        }
        Ok(branches)
    }
}

/// Truncated on-site boson basis.
#[derive(Clone, Debug)]
pub struct BosonBasis1D {
    sites: usize,
    particles: Option<usize>,
    states_per_site: usize,
    states: Vec<u128>,
}

impl BosonBasis1D {
    pub fn builder(sites: usize, states_per_site: usize) -> BosonBasisBuilder {
        BosonBasisBuilder {
            sites,
            particles: None,
            states_per_site,
        }
    }

    pub const fn sites(&self) -> usize {
        self.sites
    }

    pub const fn particles(&self) -> Option<usize> {
        self.particles
    }

    pub const fn states_per_site(&self) -> usize {
        self.states_per_site
    }
}

#[derive(Clone, Debug)]
pub struct BosonBasisBuilder {
    sites: usize,
    particles: Option<usize>,
    states_per_site: usize,
}

impl BosonBasisBuilder {
    pub const fn particles(mut self, particles: usize) -> Self {
        self.particles = Some(particles);
        self
    }

    pub fn build(self) -> Result<BosonBasis1D> {
        if self.sites == 0 || self.states_per_site == 0 {
            return Err(QuSpinError::InvalidSector(
                "boson sites and states_per_site must be positive".into(),
            ));
        }
        if self
            .particles
            .is_some_and(|count| count > self.sites * (self.states_per_site - 1))
        {
            return Err(QuSpinError::InvalidSector(
                "particle count exceeds the local cutoff".into(),
            ));
        }
        let base = u128::try_from(self.states_per_site)
            .map_err(|_| QuSpinError::InvalidSector("local cutoff is too large".into()))?;
        let exponent = u32::try_from(self.sites)
            .map_err(|_| QuSpinError::UnsupportedBackend("site count is too large".into()))?;
        let limit = base.checked_pow(exponent).ok_or_else(|| {
            QuSpinError::UnsupportedBackend("boson state encoding overflow".into())
        })?;
        let mut states = Vec::new();
        for encoded in 0..limit {
            let mut value = encoded;
            let mut total = 0_usize;
            for _ in 0..self.sites {
                total += usize::try_from(value % base).unwrap_or(usize::MAX);
                value /= base;
            }
            if self.particles.is_none_or(|count| count == total) {
                states.push(encoded);
            }
        }
        if states.is_empty() {
            return Err(QuSpinError::InvalidSector("empty boson sector".into()));
        }
        Ok(BosonBasis1D {
            sites: self.sites,
            particles: self.particles,
            states_per_site: self.states_per_site,
            states,
        })
    }
}

impl Basis for BosonBasis1D {
    type State = u128;

    fn len(&self) -> usize {
        self.states.len()
    }

    fn state(&self, index: usize) -> Result<Self::State> {
        self.states
            .get(index)
            .copied()
            .ok_or(QuSpinError::StateNotInBasis)
    }

    fn index(&self, state: Self::State) -> Result<usize> {
        state_index(&self.states, state)
    }

    fn apply_local(
        &self,
        mut state: Self::State,
        operator: &str,
        sites: &[usize],
    ) -> Result<Option<(Self::State, Complex64)>> {
        let chars = operator_chars(operator, sites)?;
        let base = self.states_per_site as u128;
        let mut amplitude = Complex64::new(1.0, 0.0);
        for (&site, op) in sites.iter().zip(chars).rev() {
            checked_site(site, self.sites)?;
            let place = base.pow(u32::try_from(site).unwrap_or(u32::MAX));
            let occupation = (state / place) % base;
            match op {
                'I' => {}
                'n' => amplitude *= occupation as f64,
                '+' if occupation + 1 < base => {
                    state += place;
                    amplitude *= ((occupation + 1) as f64).sqrt();
                }
                '-' if occupation > 0 => {
                    state -= place;
                    amplitude *= (occupation as f64).sqrt();
                }
                '+' | '-' => return Ok(None),
                _ => return Err(QuSpinError::InvalidOperator(op.to_string())),
            }
        }
        Ok(Some((state, amplitude)))
    }
}

/// Single-flavor fermion basis.
#[derive(Clone, Debug)]
pub struct SpinlessFermionBasis1D {
    sites: usize,
    particles: Option<usize>,
    momentum: Option<usize>,
    orbit_lengths: Vec<usize>,
    symmetry_lookup: HashMap<u128, SymmetryImage>,
    states: Vec<u128>,
}

impl SpinlessFermionBasis1D {
    pub fn builder(sites: usize) -> SpinlessFermionBasisBuilder {
        SpinlessFermionBasisBuilder {
            sites,
            particles: None,
            momentum: None,
        }
    }

    pub const fn sites(&self) -> usize {
        self.sites
    }

    pub const fn particles(&self) -> Option<usize> {
        self.particles
    }

    pub const fn momentum(&self) -> Option<usize> {
        self.momentum
    }
}

#[derive(Clone, Debug)]
pub struct SpinlessFermionBasisBuilder {
    sites: usize,
    particles: Option<usize>,
    momentum: Option<i32>,
}

impl SpinlessFermionBasisBuilder {
    pub const fn particles(mut self, particles: usize) -> Self {
        self.particles = Some(particles);
        self
    }

    pub const fn momentum(mut self, momentum: i32) -> Self {
        self.momentum = Some(momentum);
        self
    }

    pub fn build(self) -> Result<SpinlessFermionBasis1D> {
        let parent_states = fixed_weight_states(self.sites, self.particles)?;
        let (states, orbit_lengths, symmetry_lookup, momentum) =
            fermion_translation_sector(parent_states, self.sites, self.momentum)?;
        Ok(SpinlessFermionBasis1D {
            sites: self.sites,
            particles: self.particles,
            momentum,
            orbit_lengths,
            symmetry_lookup,
            states,
        })
    }
}

fn apply_fermion(mut state: u128, orbital: usize, op: char) -> Result<Option<(u128, Complex64)>> {
    let mask = 1_u128 << orbital;
    let occupied = state & mask != 0;
    let prior_mask = mask - 1;
    let sign = if (state & prior_mask).count_ones() % 2 == 0 {
        1.0
    } else {
        -1.0
    };
    let amplitude = match op {
        'I' => 1.0,
        'n' => return Ok(occupied.then_some((state, Complex64::new(1.0, 0.0)))),
        'z' => {
            return Ok(Some((
                state,
                Complex64::new(if occupied { 1.0 } else { -1.0 }, 0.0),
            )));
        }
        '+' if !occupied => {
            state |= mask;
            sign
        }
        '-' if occupied => {
            state &= !mask;
            sign
        }
        '+' | '-' => return Ok(None),
        _ => return Err(QuSpinError::InvalidOperator(op.to_string())),
    };
    Ok(Some((state, Complex64::new(amplitude, 0.0))))
}

impl Basis for SpinlessFermionBasis1D {
    type State = u128;

    fn len(&self) -> usize {
        self.states.len()
    }

    fn state(&self, index: usize) -> Result<Self::State> {
        self.states
            .get(index)
            .copied()
            .ok_or(QuSpinError::StateNotInBasis)
    }

    fn index(&self, state: Self::State) -> Result<usize> {
        state_index(&self.states, state)
    }

    fn apply_local(
        &self,
        mut state: Self::State,
        operator: &str,
        sites: &[usize],
    ) -> Result<Option<(Self::State, Complex64)>> {
        let source_state = state;
        let chars = operator_chars(operator, sites)?;
        let mut amplitude = Complex64::new(1.0, 0.0);
        for (&site, op) in sites.iter().zip(chars).rev() {
            checked_site(site, self.sites)?;
            let Some((next, local)) = apply_fermion(state, site, op)? else {
                return Ok(None);
            };
            state = next;
            amplitude *= local;
        }
        if self.momentum.is_some() {
            let source_index = self.index(source_state)?;
            let source_orbit = self.orbit_lengths[source_index];
            let Some(image) = self.symmetry_lookup.get(&state) else {
                return Ok(None);
            };
            amplitude *=
                (source_orbit as f64 / image.orbit_size as f64).sqrt() * image.phase.conj();
            state = image.representative;
        }
        Ok(Some((state, amplitude)))
    }
}

/// Two-flavor fermion basis with all up orbitals ordered before all down orbitals.
#[derive(Clone, Debug)]
pub struct SpinfulFermionBasis1D {
    sites: usize,
    particles_up: Option<usize>,
    particles_down: Option<usize>,
    states: Vec<u128>,
}

impl SpinfulFermionBasis1D {
    pub fn builder(sites: usize) -> SpinfulFermionBasisBuilder {
        SpinfulFermionBasisBuilder {
            sites,
            particles_up: None,
            particles_down: None,
        }
    }

    pub const fn sites(&self) -> usize {
        self.sites
    }

    pub const fn particles_up(&self) -> Option<usize> {
        self.particles_up
    }

    pub const fn particles_down(&self) -> Option<usize> {
        self.particles_down
    }
}

#[derive(Clone, Debug)]
pub struct SpinfulFermionBasisBuilder {
    sites: usize,
    particles_up: Option<usize>,
    particles_down: Option<usize>,
}

impl SpinfulFermionBasisBuilder {
    pub const fn particles_up(mut self, particles: usize) -> Self {
        self.particles_up = Some(particles);
        self
    }

    pub const fn particles_down(mut self, particles: usize) -> Self {
        self.particles_down = Some(particles);
        self
    }

    pub const fn particles(mut self, up: usize, down: usize) -> Self {
        self.particles_up = Some(up);
        self.particles_down = Some(down);
        self
    }

    pub fn build(self) -> Result<SpinfulFermionBasis1D> {
        if self.sites > 64 {
            return Err(QuSpinError::UnsupportedBackend(
                "the packed spinful backend supports at most 64 sites".into(),
            ));
        }
        let up_states = fixed_weight_states(self.sites, self.particles_up)?;
        let down_states = fixed_weight_states(self.sites, self.particles_down)?;
        let mut states = Vec::with_capacity(up_states.len() * down_states.len());
        for down in down_states {
            for &up in &up_states {
                states.push(up | (down << self.sites));
            }
        }
        states.sort_unstable();
        Ok(SpinfulFermionBasis1D {
            sites: self.sites,
            particles_up: self.particles_up,
            particles_down: self.particles_down,
            states,
        })
    }
}

impl Basis for SpinfulFermionBasis1D {
    type State = u128;

    fn len(&self) -> usize {
        self.states.len()
    }

    fn state(&self, index: usize) -> Result<Self::State> {
        self.states
            .get(index)
            .copied()
            .ok_or(QuSpinError::StateNotInBasis)
    }

    fn index(&self, state: Self::State) -> Result<usize> {
        state_index(&self.states, state)
    }

    fn apply_local(
        &self,
        mut state: Self::State,
        operator: &str,
        sites: &[usize],
    ) -> Result<Option<(Self::State, Complex64)>> {
        let chars = operator_chars(operator, sites)?;
        let split = operator
            .find('|')
            .map_or(chars.len(), |position| operator[..position].chars().count());
        let mut amplitude = Complex64::new(1.0, 0.0);
        for (position, (&site, op)) in sites.iter().zip(chars).enumerate().rev() {
            checked_site(site, self.sites)?;
            let orbital = if position < split {
                site
            } else {
                self.sites + site
            };
            let Some((next, local)) = apply_fermion(state, orbital, op)? else {
                return Ok(None);
            };
            state = next;
            amplitude *= local;
        }
        Ok(Some((state, amplitude)))
    }
}

type UserAction<State> =
    Arc<dyn Fn(State, usize) -> Result<Option<(State, Complex64)>> + Send + Sync>;

/// Callback-defined constrained basis using the same assembly path as built-ins.
pub struct UserBasis<State>
where
    State: Copy + Eq + Hash + Send + Sync,
{
    sites: usize,
    states: Vec<State>,
    indices: HashMap<State, usize>,
    operators: HashMap<char, UserAction<State>>,
}

impl<State> UserBasis<State>
where
    State: Copy + Eq + Hash + Send + Sync,
{
    pub fn builder(sites: usize) -> UserBasisBuilder<State> {
        UserBasisBuilder {
            sites,
            states: Vec::new(),
            operators: HashMap::new(),
        }
    }
}

pub struct UserBasisBuilder<State>
where
    State: Copy + Eq + Hash + Send + Sync,
{
    sites: usize,
    states: Vec<State>,
    operators: HashMap<char, UserAction<State>>,
}

impl<State> UserBasisBuilder<State>
where
    State: Copy + Eq + Hash + Send + Sync + 'static,
{
    pub fn states(mut self, states: impl IntoIterator<Item = State>) -> Self {
        self.states = states.into_iter().collect();
        self
    }

    pub fn operator<F>(mut self, name: char, action: F) -> Self
    where
        F: Fn(State, usize) -> Result<Option<(State, Complex64)>> + Send + Sync + 'static,
    {
        self.operators.insert(name, Arc::new(action));
        self
    }

    pub fn build(self) -> Result<UserBasis<State>> {
        if self.states.is_empty() {
            return Err(QuSpinError::InvalidSector(
                "UserBasis requires at least one accepted state".into(),
            ));
        }
        let mut indices = HashMap::with_capacity(self.states.len());
        for (index, state) in self.states.iter().copied().enumerate() {
            if indices.insert(state, index).is_some() {
                return Err(QuSpinError::InvalidSector(
                    "UserBasis states must be unique".into(),
                ));
            }
        }
        Ok(UserBasis {
            sites: self.sites,
            states: self.states,
            indices,
            operators: self.operators,
        })
    }
}

impl UserBasisBuilder<u128> {
    pub fn state_filter<F>(mut self, keep: F) -> Result<Self>
    where
        F: Fn(u128) -> bool,
    {
        if self.sites > 127 {
            return Err(QuSpinError::UnsupportedBackend(
                "u128 UserBasis filters support at most 127 sites".into(),
            ));
        }
        let limit = 1_u128 << self.sites;
        self.states = (0..limit).filter(|state| keep(*state)).collect();
        Ok(self)
    }
}

impl<State> Basis for UserBasis<State>
where
    State: Copy + Eq + Hash + Send + Sync + 'static,
{
    type State = State;

    fn len(&self) -> usize {
        self.states.len()
    }

    fn state(&self, index: usize) -> Result<Self::State> {
        self.states
            .get(index)
            .copied()
            .ok_or(QuSpinError::StateNotInBasis)
    }

    fn index(&self, state: Self::State) -> Result<usize> {
        self.indices
            .get(&state)
            .copied()
            .ok_or(QuSpinError::StateNotInBasis)
    }

    fn apply_local(
        &self,
        mut state: Self::State,
        operator: &str,
        sites: &[usize],
    ) -> Result<Option<(Self::State, Complex64)>> {
        let chars = operator_chars(operator, sites)?;
        let mut amplitude = Complex64::new(1.0, 0.0);
        for (&site, op) in sites.iter().zip(chars).rev() {
            checked_site(site, self.sites)?;
            let action = self
                .operators
                .get(&op)
                .ok_or_else(|| QuSpinError::InvalidOperator(op.to_string()))?;
            let Some((next, local)) = action(state, site)? else {
                return Ok(None);
            };
            state = next;
            amplitude *= local;
        }
        Ok(Some((state, amplitude)))
    }
}

/// Finite symmetry action, including any phase acquired by the state.
pub trait SymmetryMap<State>: Send + Sync {
    fn period(&self) -> usize;
    fn apply(&self, state: State) -> Result<(State, Complex64)>;
}

type SymmetryAction<State> = Arc<dyn Fn(State) -> Result<(State, Complex64)> + Send + Sync>;

/// Closure-backed finite map for lattice, particle-hole, or user symmetries.
pub struct ClosureSymmetryMap<State> {
    period: usize,
    action: SymmetryAction<State>,
}

impl<State> ClosureSymmetryMap<State> {
    pub fn new<F>(period: usize, action: F) -> Result<Self>
    where
        F: Fn(State) -> Result<(State, Complex64)> + Send + Sync + 'static,
    {
        if period == 0 {
            return Err(QuSpinError::InvalidSector(
                "a symmetry-map period must be positive".into(),
            ));
        }
        Ok(Self {
            period,
            action: Arc::new(action),
        })
    }
}

impl<State> SymmetryMap<State> for ClosureSymmetryMap<State>
where
    State: Copy,
{
    fn period(&self) -> usize {
        self.period
    }

    fn apply(&self, state: State) -> Result<(State, Complex64)> {
        (self.action)(state)
    }
}

struct SymmetryGenerator<State> {
    map: Arc<dyn SymmetryMap<State>>,
    sector: i32,
}

/// Commuting finite maps and their one-dimensional character sectors.
pub struct SymmetrySector<State> {
    generators: Vec<SymmetryGenerator<State>>,
}

impl<State> SymmetrySector<State> {
    pub fn new() -> Self {
        Self {
            generators: Vec::new(),
        }
    }

    pub fn with_map<M>(mut self, map: M, sector: i32) -> Self
    where
        M: SymmetryMap<State> + 'static,
    {
        self.generators.push(SymmetryGenerator {
            map: Arc::new(map),
            sector,
        });
        self
    }

    pub fn generators(&self) -> usize {
        self.generators.len()
    }
}

impl<State> Default for SymmetrySector<State> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, Debug)]
struct GeneralSymmetryImage<State> {
    representative: State,
    phase: Complex64,
    orbit_size: usize,
}

fn enumerate_symmetry_images<State>(
    state: State,
    generators: &[SymmetryGenerator<State>],
) -> Result<HashMap<State, Complex64>>
where
    State: Copy + Eq + Hash,
{
    fn visit<State>(
        generator_index: usize,
        state: State,
        amplitude: Complex64,
        generators: &[SymmetryGenerator<State>],
        output: &mut HashMap<State, Complex64>,
    ) -> Result<()>
    where
        State: Copy + Eq + Hash,
    {
        if generator_index == generators.len() {
            *output.entry(state).or_insert(Complex64::new(0.0, 0.0)) += amplitude;
            return Ok(());
        }
        let generator = &generators[generator_index];
        let period = generator.map.period();
        if period == 0 {
            return Err(QuSpinError::InvalidSector(
                "a symmetry-map period must be positive".into(),
            ));
        }
        let normalized_sector = i64::from(generator.sector).rem_euclid(period as i64) as usize;
        let mut image = state;
        let mut map_phase = Complex64::new(1.0, 0.0);
        for power in 0..period {
            let angle = -std::f64::consts::TAU * (normalized_sector * power) as f64 / period as f64;
            visit(
                generator_index + 1,
                image,
                amplitude * map_phase * Complex64::from_polar(1.0, angle),
                generators,
                output,
            )?;
            let (next, phase) = generator.map.apply(image)?;
            if !phase.re.is_finite() || !phase.im.is_finite() {
                return Err(QuSpinError::IncompatibleSymmetry(
                    "a symmetry map returned a non-finite phase".into(),
                ));
            }
            image = next;
            map_phase *= phase;
        }
        if image != state || (map_phase - Complex64::new(1.0, 0.0)).norm() > 1.0e-10 {
            return Err(QuSpinError::IncompatibleSymmetry(
                "a symmetry map does not close at its declared period".into(),
            ));
        }
        Ok(())
    }

    let mut images = HashMap::new();
    visit(0, state, Complex64::new(1.0, 0.0), generators, &mut images)?;
    Ok(images)
}

/// Arbitrary finite-map reduction of any concrete parent basis.
pub struct GeneralBasis<Parent>
where
    Parent: Basis,
    Parent::State: Hash + Ord,
{
    parent: Parent,
    states: Vec<Parent::State>,
    orbit_lengths: Vec<usize>,
    lookup: HashMap<Parent::State, GeneralSymmetryImage<Parent::State>>,
}

impl<Parent> GeneralBasis<Parent>
where
    Parent: Basis,
    Parent::State: Hash + Ord,
{
    pub fn new(parent: Parent, sector: SymmetrySector<Parent::State>) -> Result<Self> {
        if sector.generators.is_empty() {
            let mut states = Vec::with_capacity(parent.len());
            let mut lookup = HashMap::with_capacity(parent.len());
            for index in 0..parent.len() {
                let state = parent.state(index)?;
                states.push(state);
                lookup.insert(
                    state,
                    GeneralSymmetryImage {
                        representative: state,
                        phase: Complex64::new(1.0, 0.0),
                        orbit_size: 1,
                    },
                );
            }
            states.sort_unstable();
            return Ok(Self {
                parent,
                orbit_lengths: vec![1; states.len()],
                states,
                lookup,
            });
        }

        let mut visited = HashSet::with_capacity(parent.len());
        let mut representatives = Vec::new();
        let mut lookup = HashMap::with_capacity(parent.len());
        for index in 0..parent.len() {
            let seed = parent.state(index)?;
            if visited.contains(&seed) {
                continue;
            }
            let mut coefficients = enumerate_symmetry_images(seed, &sector.generators)?;
            for state in coefficients.keys() {
                parent.index(*state).map_err(|_| {
                    QuSpinError::IncompatibleSymmetry(
                        "a symmetry map leaves the parent basis".into(),
                    )
                })?;
                visited.insert(*state);
            }
            coefficients.retain(|_, coefficient| coefficient.norm() > 1.0e-12);
            if coefficients.is_empty() {
                continue;
            }
            let representative = *coefficients.keys().min().ok_or_else(|| {
                QuSpinError::InvalidSector("symmetry projection generated no state".into())
            })?;
            let representative_coefficient = coefficients[&representative];
            let gauge = representative_coefficient / representative_coefficient.norm();
            let norm = coefficients
                .values()
                .map(Complex64::norm_sqr)
                .sum::<f64>()
                .sqrt();
            let orbit_size = coefficients.len();
            let expected_magnitude = 1.0 / (orbit_size as f64).sqrt();
            for (&state, &coefficient) in &coefficients {
                let normalized = coefficient / (gauge * norm);
                if (normalized.norm() - expected_magnitude).abs() > 1.0e-10 {
                    return Err(QuSpinError::IncompatibleSymmetry(
                        "symmetry maps do not define a one-dimensional orbit sector".into(),
                    ));
                }
                lookup.insert(
                    state,
                    GeneralSymmetryImage {
                        representative,
                        phase: normalized / expected_magnitude,
                        orbit_size,
                    },
                );
            }
            representatives.push((representative, orbit_size));
        }
        representatives.sort_by_key(|(state, _)| *state);
        if representatives.is_empty() {
            return Err(QuSpinError::InvalidSector(
                "the requested general symmetry sector is empty".into(),
            ));
        }
        let (states, orbit_lengths) = representatives.into_iter().unzip();
        Ok(Self {
            parent,
            states,
            orbit_lengths,
            lookup,
        })
    }

    pub fn parent(&self) -> &Parent {
        &self.parent
    }
}

impl<Parent> Basis for GeneralBasis<Parent>
where
    Parent: Basis,
    Parent::State: Hash + Ord + 'static,
{
    type State = Parent::State;

    fn len(&self) -> usize {
        self.states.len()
    }

    fn state(&self, index: usize) -> Result<Self::State> {
        self.states
            .get(index)
            .copied()
            .ok_or(QuSpinError::StateNotInBasis)
    }

    fn index(&self, state: Self::State) -> Result<usize> {
        self.states
            .binary_search(&state)
            .map_err(|_| QuSpinError::StateNotInBasis)
    }

    fn apply_local(
        &self,
        state: Self::State,
        operator: &str,
        sites: &[usize],
    ) -> Result<Option<(Self::State, Complex64)>> {
        let transitions = self.apply_local_transitions(state, operator, sites)?;
        match transitions.as_slice() {
            [] => Ok(None),
            [transition] => Ok(Some(*transition)),
            _ => Err(QuSpinError::UnsupportedBackend(
                "this reduced local action branches; use apply_local_transitions".into(),
            )),
        }
    }

    fn apply_local_transitions(
        &self,
        state: Self::State,
        operator: &str,
        sites: &[usize],
    ) -> Result<Vec<(Self::State, Complex64)>> {
        let source_index = self.index(state)?;
        let source_orbit = self.orbit_lengths[source_index];
        let mut reduced = HashMap::<Self::State, Complex64>::new();
        for (target, mut amplitude) in self
            .parent
            .apply_local_transitions(state, operator, sites)?
        {
            let Some(image) = self.lookup.get(&target) else {
                continue;
            };
            amplitude *=
                (source_orbit as f64 / image.orbit_size as f64).sqrt() * image.phase.conj();
            *reduced
                .entry(image.representative)
                .or_insert(Complex64::new(0.0, 0.0)) += amplitude;
        }
        let mut transitions: Vec<_> = reduced
            .into_iter()
            .filter(|(_, amplitude)| amplitude.norm() > f64::EPSILON)
            .collect();
        transitions.sort_by_key(|(state, _)| *state);
        Ok(transitions)
    }
}

pub type SpinBasisGeneral = GeneralBasis<SpinBasis1D>;
pub type BosonBasisGeneral = GeneralBasis<BosonBasis1D>;
pub type SpinlessFermionBasisGeneral = GeneralBasis<SpinlessFermionBasis1D>;
pub type SpinfulFermionBasisGeneral = GeneralBasis<SpinfulFermionBasis1D>;

/// Direct-product basis. Operator strings use `left|right` factor syntax.
#[derive(Clone, Debug)]
pub struct TensorBasis<Left, Right> {
    left: Left,
    right: Right,
}

impl<Left, Right> TensorBasis<Left, Right>
where
    Left: Basis,
    Right: Basis,
{
    pub fn new(left: Left, right: Right) -> Result<Self> {
        left.len()
            .checked_mul(right.len())
            .ok_or_else(|| QuSpinError::UnsupportedBackend("tensor-basis size overflow".into()))?;
        Ok(Self { left, right })
    }

    pub fn left(&self) -> &Left {
        &self.left
    }

    pub fn right(&self) -> &Right {
        &self.right
    }
}

impl<Left, Right> Basis for TensorBasis<Left, Right>
where
    Left: Basis,
    Right: Basis,
    Left::State: 'static,
    Right::State: 'static,
{
    type State = (Left::State, Right::State);

    fn len(&self) -> usize {
        self.left.len() * self.right.len()
    }

    fn state(&self, index: usize) -> Result<Self::State> {
        if index >= self.len() {
            return Err(QuSpinError::StateNotInBasis);
        }
        Ok((
            self.left.state(index / self.right.len())?,
            self.right.state(index % self.right.len())?,
        ))
    }

    fn index(&self, state: Self::State) -> Result<usize> {
        Ok(self.left.index(state.0)? * self.right.len() + self.right.index(state.1)?)
    }

    fn apply_local(
        &self,
        state: Self::State,
        operator: &str,
        sites: &[usize],
    ) -> Result<Option<(Self::State, Complex64)>> {
        let transitions = self.apply_local_transitions(state, operator, sites)?;
        match transitions.as_slice() {
            [] => Ok(None),
            [transition] => Ok(Some(*transition)),
            _ => Err(QuSpinError::UnsupportedBackend(
                "this tensor local action branches; use apply_local_transitions".into(),
            )),
        }
    }

    fn apply_local_transitions(
        &self,
        state: Self::State,
        operator: &str,
        sites: &[usize],
    ) -> Result<Vec<(Self::State, Complex64)>> {
        let (left_operator, right_operator) = operator.split_once('|').ok_or_else(|| {
            QuSpinError::InvalidOperator(
                "tensor-basis operator strings must contain one `|` separator".into(),
            )
        })?;
        if right_operator.contains('|') {
            return Err(QuSpinError::InvalidOperator(
                "a two-factor tensor operator contains too many separators".into(),
            ));
        }
        let left_arity = left_operator.chars().count();
        let right_arity = right_operator.chars().count();
        if sites.len() != left_arity + right_arity {
            return Err(QuSpinError::InvalidCoupling(
                "tensor operator arity does not match its sites".into(),
            ));
        }
        let left_transitions = if left_operator.is_empty() {
            vec![(state.0, Complex64::new(1.0, 0.0))]
        } else {
            self.left
                .apply_local_transitions(state.0, left_operator, &sites[..left_arity])?
        };
        let right_transitions = if right_operator.is_empty() {
            vec![(state.1, Complex64::new(1.0, 0.0))]
        } else {
            self.right
                .apply_local_transitions(state.1, right_operator, &sites[left_arity..])?
        };
        let mut transitions = Vec::with_capacity(left_transitions.len() * right_transitions.len());
        for &(left_state, left_amplitude) in &left_transitions {
            for &(right_state, right_amplitude) in &right_transitions {
                transitions.push(((left_state, right_state), left_amplitude * right_amplitude));
            }
        }
        Ok(transitions)
    }
}

/// Matter basis tensored with one truncated photon mode, optionally at fixed
/// total excitation number.
pub struct PhotonBasis<Matter>
where
    Matter: Basis,
    Matter::State: Hash,
{
    tensor: TensorBasis<Matter, BosonBasis1D>,
    states: Vec<(Matter::State, u128)>,
    indices: HashMap<(Matter::State, u128), usize>,
    total_excitations: Option<usize>,
}

impl<Matter> PhotonBasis<Matter>
where
    Matter: Basis,
    Matter::State: Hash + 'static,
{
    pub fn new(matter: Matter, photon: BosonBasis1D) -> Result<Self> {
        Self::build(matter, photon, None, |_| 0)
    }

    pub fn fixed_total_excitations<F>(
        matter: Matter,
        photon: BosonBasis1D,
        total: usize,
        matter_excitations: F,
    ) -> Result<Self>
    where
        F: Fn(Matter::State) -> usize,
    {
        Self::build(matter, photon, Some(total), matter_excitations)
    }

    fn build<F>(
        matter: Matter,
        photon: BosonBasis1D,
        total_excitations: Option<usize>,
        matter_excitations: F,
    ) -> Result<Self>
    where
        F: Fn(Matter::State) -> usize,
    {
        if photon.sites() != 1 {
            return Err(QuSpinError::InvalidSector(
                "PhotonBasis requires a one-mode boson basis".into(),
            ));
        }
        let tensor = TensorBasis::new(matter, photon)?;
        let mut states = Vec::new();
        for index in 0..tensor.len() {
            let state = tensor.state(index)?;
            if total_excitations
                .is_none_or(|total| matter_excitations(state.0) + state.1 as usize == total)
            {
                states.push(state);
            }
        }
        if states.is_empty() {
            return Err(QuSpinError::InvalidSector(
                "the requested photon sector is empty".into(),
            ));
        }
        let indices = states
            .iter()
            .copied()
            .enumerate()
            .map(|(index, state)| (state, index))
            .collect();
        Ok(Self {
            tensor,
            states,
            indices,
            total_excitations,
        })
    }

    pub const fn total_excitations(&self) -> Option<usize> {
        self.total_excitations
    }

    pub fn matter(&self) -> &Matter {
        self.tensor.left()
    }

    pub fn photon(&self) -> &BosonBasis1D {
        self.tensor.right()
    }
}

impl<Matter> Basis for PhotonBasis<Matter>
where
    Matter: Basis,
    Matter::State: Hash + 'static,
{
    type State = (Matter::State, u128);

    fn len(&self) -> usize {
        self.states.len()
    }

    fn state(&self, index: usize) -> Result<Self::State> {
        self.states
            .get(index)
            .copied()
            .ok_or(QuSpinError::StateNotInBasis)
    }

    fn index(&self, state: Self::State) -> Result<usize> {
        self.indices
            .get(&state)
            .copied()
            .ok_or(QuSpinError::StateNotInBasis)
    }

    fn apply_local(
        &self,
        state: Self::State,
        operator: &str,
        sites: &[usize],
    ) -> Result<Option<(Self::State, Complex64)>> {
        let transitions = self.apply_local_transitions(state, operator, sites)?;
        match transitions.as_slice() {
            [] => Ok(None),
            [transition] => Ok(Some(*transition)),
            _ => Err(QuSpinError::UnsupportedBackend(
                "this photon-basis action branches; use apply_local_transitions".into(),
            )),
        }
    }

    fn apply_local_transitions(
        &self,
        state: Self::State,
        operator: &str,
        sites: &[usize],
    ) -> Result<Vec<(Self::State, Complex64)>> {
        Ok(self
            .tensor
            .apply_local_transitions(state, operator, sites)?
            .into_iter()
            .filter(|(target, _)| self.indices.contains_key(target))
            .collect())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StateStorage {
    U128,
    U256,
    U1024,
    U4096,
    U16384,
}

/// Fixed-width state used by wide user and general bases.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct WideState<const WORDS: usize> {
    words: [u64; WORDS],
}

impl<const WORDS: usize> WideState<WORDS> {
    pub const fn zero() -> Self {
        Self { words: [0; WORDS] }
    }

    pub const fn capacity_bits() -> usize {
        WORDS * 64
    }

    pub fn from_words(words: [u64; WORDS]) -> Self {
        Self { words }
    }

    pub fn words(&self) -> &[u64; WORDS] {
        &self.words
    }

    pub fn bit(&self, index: usize) -> Result<bool> {
        if index >= Self::capacity_bits() {
            return Err(QuSpinError::InvalidSite {
                site: index,
                sites: Self::capacity_bits(),
            });
        }
        Ok(self.words[index / 64] & (1_u64 << (index % 64)) != 0)
    }

    pub fn with_bit(mut self, index: usize, occupied: bool) -> Result<Self> {
        if index >= Self::capacity_bits() {
            return Err(QuSpinError::InvalidSite {
                site: index,
                sites: Self::capacity_bits(),
            });
        }
        let mask = 1_u64 << (index % 64);
        if occupied {
            self.words[index / 64] |= mask;
        } else {
            self.words[index / 64] &= !mask;
        }
        Ok(self)
    }

    pub fn count_ones(&self) -> usize {
        self.words
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum()
    }

    pub fn bitwise_and(self, right: Self) -> Self {
        Self::from_words(std::array::from_fn(|index| {
            self.words[index] & right.words[index]
        }))
    }

    pub fn bitwise_or(self, right: Self) -> Self {
        Self::from_words(std::array::from_fn(|index| {
            self.words[index] | right.words[index]
        }))
    }

    pub fn bitwise_xor(self, right: Self) -> Self {
        Self::from_words(std::array::from_fn(|index| {
            self.words[index] ^ right.words[index]
        }))
    }

    pub fn bitwise_not(self) -> Self {
        Self::from_words(std::array::from_fn(|index| !self.words[index]))
    }

    pub fn left_shift(self, shift: usize) -> Self {
        if shift >= Self::capacity_bits() {
            return Self::zero();
        }
        let word_shift = shift / 64;
        let bit_shift = shift % 64;
        let mut words = [0_u64; WORDS];
        for target in (word_shift..WORDS).rev() {
            let source = target - word_shift;
            words[target] |= self.words[source] << bit_shift;
            if bit_shift > 0 && source > 0 {
                words[target] |= self.words[source - 1] >> (64 - bit_shift);
            }
        }
        Self::from_words(words)
    }

    pub fn right_shift(self, shift: usize) -> Self {
        if shift >= Self::capacity_bits() {
            return Self::zero();
        }
        let word_shift = shift / 64;
        let bit_shift = shift % 64;
        let mut words = [0_u64; WORDS];
        for (target, word) in words.iter_mut().enumerate().take(WORDS - word_shift) {
            let source = target + word_shift;
            *word |= self.words[source] >> bit_shift;
            if bit_shift > 0 && source + 1 < WORDS {
                *word |= self.words[source + 1] << (64 - bit_shift);
            }
        }
        Self::from_words(words)
    }
}

pub type U256 = WideState<4>;
pub type U1024 = WideState<16>;
pub type U4096 = WideState<64>;
pub type U16384 = WideState<256>;
pub type UInt256 = U256;
pub type UInt1024 = U1024;
pub type UInt4096 = U4096;
pub type UInt16384 = U16384;

pub fn basis_zeros<const WORDS: usize>(length: usize) -> Vec<WideState<WORDS>> {
    vec![WideState::zero(); length]
}

pub fn basis_ones<const WORDS: usize>(length: usize) -> Vec<WideState<WORDS>> {
    vec![WideState::zero().bitwise_not(); length]
}

pub fn bitwise_and<const WORDS: usize>(
    left: WideState<WORDS>,
    right: WideState<WORDS>,
) -> WideState<WORDS> {
    left.bitwise_and(right)
}

pub fn bitwise_or<const WORDS: usize>(
    left: WideState<WORDS>,
    right: WideState<WORDS>,
) -> WideState<WORDS> {
    left.bitwise_or(right)
}

pub fn bitwise_xor<const WORDS: usize>(
    left: WideState<WORDS>,
    right: WideState<WORDS>,
) -> WideState<WORDS> {
    left.bitwise_xor(right)
}

pub fn bitwise_not<const WORDS: usize>(value: WideState<WORDS>) -> WideState<WORDS> {
    value.bitwise_not()
}

pub fn bitwise_leftshift<const WORDS: usize>(
    value: WideState<WORDS>,
    shift: usize,
) -> WideState<WORDS> {
    value.left_shift(shift)
}

pub fn bitwise_rightshift<const WORDS: usize>(
    value: WideState<WORDS>,
    shift: usize,
) -> WideState<WORDS> {
    value.right_shift(shift)
}

pub fn python_int_to_basis_int<const WORDS: usize>(value: u128) -> WideState<WORDS> {
    let mut words = [0_u64; WORDS];
    if WORDS > 0 {
        words[0] = value as u64;
    }
    if WORDS > 1 {
        words[1] = (value >> 64) as u64;
    }
    WideState::from_words(words)
}

pub fn basis_int_to_python_int<const WORDS: usize>(value: WideState<WORDS>) -> Result<u128> {
    if value.words.iter().skip(2).any(|word| *word != 0) {
        return Err(QuSpinError::UnsupportedBackend(
            "wide basis integer does not fit into a Python-compatible u128".into(),
        ));
    }
    Ok(u128::from(value.words.first().copied().unwrap_or_default())
        | (u128::from(value.words.get(1).copied().unwrap_or_default()) << 64))
}

pub fn get_basis_type(
    sites: usize,
    _particles: Option<usize>,
    states_per_site: usize,
) -> Result<StateStorage> {
    if states_per_site < 2 {
        return Err(QuSpinError::InvalidSector(
            "states_per_site must be at least two".into(),
        ));
    }
    let bits_per_site =
        usize::try_from(usize::BITS - (states_per_site - 1).leading_zeros()).unwrap_or(usize::MAX);
    let bits = sites
        .checked_mul(bits_per_site)
        .ok_or_else(|| QuSpinError::UnsupportedBackend("basis bit width overflow".into()))?;
    match bits {
        0..=128 => Ok(StateStorage::U128),
        129..=256 => Ok(StateStorage::U256),
        257..=1024 => Ok(StateStorage::U1024),
        1025..=4096 => Ok(StateStorage::U4096),
        4097..=16384 => Ok(StateStorage::U16384),
        _ => Err(QuSpinError::UnsupportedBackend(
            "basis requires more than 16384 state bits".into(),
        )),
    }
}

pub fn coherent_state(amplitude: Complex64, states: usize) -> Result<Vec<Complex64>> {
    if states == 0 || !amplitude.re.is_finite() || !amplitude.im.is_finite() {
        return Err(QuSpinError::InvalidOptions(
            "coherent-state amplitude must be finite and the cutoff positive".into(),
        ));
    }
    let mut coefficients = Vec::with_capacity(states);
    let mut coefficient = Complex64::new((-0.5 * amplitude.norm_sqr()).exp(), 0.0);
    coefficients.push(coefficient);
    for occupation in 1..states {
        coefficient *= amplitude / (occupation as f64).sqrt();
        coefficients.push(coefficient);
    }
    Ok(coefficients)
}

fn binomial(n: usize, k: usize) -> usize {
    let k = k.min(n.saturating_sub(k));
    (0..k).fold(1_usize, |value, index| {
        value.saturating_mul(n - index) / (index + 1)
    })
}

/// Dimension of a spin-half chain plus one photon mode at fixed excitation.
pub fn photon_hspace_dim(
    sites: usize,
    total_excitations: Option<usize>,
    photon_cutoff: Option<usize>,
) -> Result<usize> {
    match (total_excitations, photon_cutoff) {
        (None, Some(cutoff)) => 1_usize
            .checked_shl(u32::try_from(sites).unwrap_or(u32::MAX))
            .and_then(|matter| matter.checked_mul(cutoff.saturating_add(1)))
            .ok_or_else(|| QuSpinError::UnsupportedBackend("photon dimension overflow".into())),
        (Some(total), cutoff) => {
            let minimum_matter =
                cutoff.map_or(0, |maximum_photons| total.saturating_sub(maximum_photons));
            let maximum_matter = sites.min(total);
            Ok((minimum_matter..=maximum_matter)
                .map(|matter| binomial(sites, matter))
                .sum())
        }
        (None, None) => Err(QuSpinError::InvalidSector(
            "either total excitation or photon cutoff must be finite".into(),
        )),
    }
}

/// Sparse isometric lift from a symmetry-reduced basis to its parent basis.
#[derive(Clone, Debug)]
pub struct BasisProjector {
    source_dimension: usize,
    reduced_dimension: usize,
    column_offsets: Vec<usize>,
    row_indices: Vec<usize>,
    values: Vec<Complex64>,
}

impl BasisProjector {
    pub fn from_general<Parent>(basis: &GeneralBasis<Parent>) -> Result<Self>
    where
        Parent: Basis,
        Parent::State: Hash + Ord + 'static,
    {
        let mut by_column = vec![Vec::<(usize, Complex64)>::new(); basis.len()];
        for row in 0..basis.parent.len() {
            let state = basis.parent.state(row)?;
            let Some(image) = basis.lookup.get(&state) else {
                continue;
            };
            let column = basis.index(image.representative)?;
            by_column[column].push((row, image.phase / (image.orbit_size as f64).sqrt()));
        }
        let mut column_offsets = Vec::with_capacity(basis.len() + 1);
        let mut row_indices = Vec::new();
        let mut values = Vec::new();
        column_offsets.push(0);
        for column in &mut by_column {
            column.sort_by_key(|(row, _)| *row);
            for &(row, value) in column.iter() {
                row_indices.push(row);
                values.push(value);
            }
            column_offsets.push(row_indices.len());
        }
        Ok(Self {
            source_dimension: basis.parent.len(),
            reduced_dimension: basis.len(),
            column_offsets,
            row_indices,
            values,
        })
    }

    pub const fn source_dimension(&self) -> usize {
        self.source_dimension
    }

    pub const fn reduced_dimension(&self) -> usize {
        self.reduced_dimension
    }

    /// Apply the adjoint projector to a parent-space vector.
    pub fn project(&self, parent: &[Complex64], reduced: &mut [Complex64]) -> Result<()> {
        if parent.len() != self.source_dimension || reduced.len() != self.reduced_dimension {
            return Err(QuSpinError::DimensionMismatch(
                "projector input or output length does not match".into(),
            ));
        }
        reduced.fill(Complex64::new(0.0, 0.0));
        for (column, reduced_value) in reduced.iter_mut().enumerate() {
            for position in self.column_offsets[column]..self.column_offsets[column + 1] {
                *reduced_value += self.values[position].conj() * parent[self.row_indices[position]];
            }
        }
        Ok(())
    }
}

impl LinearOperator for BasisProjector {
    fn shape(&self) -> (usize, usize) {
        (self.source_dimension, self.reduced_dimension)
    }

    fn format(&self) -> MatrixFormat {
        MatrixFormat::Csc
    }

    fn apply(&self, input: &[Complex64], output: &mut [Complex64]) -> Result<()> {
        check_apply_shape(self.shape(), input, output)?;
        output.fill(Complex64::new(0.0, 0.0));
        for (column, &input_value) in input.iter().enumerate() {
            for position in self.column_offsets[column]..self.column_offsets[column + 1] {
                output[self.row_indices[position]] += self.values[position] * input_value;
            }
        }
        Ok(())
    }
}
