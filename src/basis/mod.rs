use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;

use num_complex::Complex64;

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

fn state_index(states: &[u128], state: u128) -> Result<usize> {
    states
        .binary_search(&state)
        .map_err(|_| QuSpinError::StateNotInBasis)
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

/// Spin-chain basis for the full or fixed-magnetization spin-one-half space.
#[derive(Clone, Debug)]
pub struct SpinBasis1D {
    sites: usize,
    spin_twice: u16,
    up: Option<usize>,
    pauli: bool,
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
        if self.spin_twice != 1 {
            return Err(QuSpinError::UnsupportedBackend(
                "the first backend supports spin one-half only".into(),
            ));
        }
        if self.momentum.is_some() || self.parity.is_some() {
            return Err(QuSpinError::UnsupportedBackend(
                "translation and parity sectors are not active yet".into(),
            ));
        }
        let states = fixed_weight_states(self.sites, self.up)?;
        if states.is_empty() {
            return Err(QuSpinError::InvalidSector("empty spin sector".into()));
        }
        Ok(SpinBasis1D {
            sites: self.sites,
            spin_twice: self.spin_twice,
            up: self.up,
            pauli: self.pauli,
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
        mut state: Self::State,
        operator: &str,
        sites: &[usize],
    ) -> Result<Option<(Self::State, Complex64)>> {
        let chars = operator_chars(operator, sites)?;
        let mut amplitude = Complex64::new(1.0, 0.0);
        for (&site, op) in sites.iter().zip(chars).rev() {
            checked_site(site, self.sites)?;
            let mask = 1_u128 << site;
            let up = state & mask != 0;
            match op {
                'I' => {}
                'z' => {
                    let value = if up { 1.0 } else { -1.0 };
                    amplitude *= if self.pauli { value } else { value * 0.5 };
                }
                '+' if !up => state |= mask,
                '-' if up => state &= !mask,
                '+' | '-' => return Ok(None),
                'x' => {
                    state ^= mask;
                    if !self.pauli {
                        amplitude *= 0.5;
                    }
                }
                'y' => {
                    state ^= mask;
                    let value = if up {
                        Complex64::new(0.0, 1.0)
                    } else {
                        Complex64::new(0.0, -1.0)
                    };
                    amplitude *= if self.pauli { value } else { value * 0.5 };
                }
                _ => return Err(QuSpinError::InvalidOperator(op.to_string())),
            }
        }
        Ok(Some((state, amplitude)))
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
        if self.momentum.is_some() {
            return Err(QuSpinError::UnsupportedBackend(
                "fermion momentum sectors are not active yet".into(),
            ));
        }
        let states = fixed_weight_states(self.sites, self.particles)?;
        Ok(SpinlessFermionBasis1D {
            sites: self.sites,
            particles: self.particles,
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
