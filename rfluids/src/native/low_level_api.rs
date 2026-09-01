use core::ffi::c_long;
use std::{borrow::Cow, marker::PhantomData};

use coolprop_sys::{COOLPROP, bindings};

use super::{
    CoolPropError, Result,
    common::{
        ErrorBuffer, PhantomUnsync, c_string_trimmed, factory_requires_exclusive,
        state_requires_exclusive,
    },
};
use crate::substance::{Substance, SubstanceWithBackend};

/// Default `max_components` for [`AbstractState::mole_fractions`] and
/// [`AbstractState::mole_fractions_sat_state`], generous above any real-world mixture, for
/// callers with no better estimate of the component count to pass.
///
/// `CoolProp`'s C API requires a caller-preallocated buffer sized up front (see
/// `AbstractState_get_mole_fractions`'s `maxN`). Prefer passing the actual expected component
/// count when it's known (e.g. from a [`Substance`]) -- it catches a mismatch as a normal
/// [`CoolPropError::TooManyComponents`] instead of silently allocating more than needed.
pub const MAX_COMPONENTS: usize = 64;

/// `CoolProp` thread safe low-level API.
#[derive(Debug)]
pub struct AbstractState {
    handle: c_long,
    exclusive: bool,
    marker: PhantomUnsync,
}

/// Phase-envelope trace read back by
/// [`AbstractState::phase_envelope_data`], in `CoolProp`'s native SI units.
///
/// `liquid_mole_fractions[k]`/`vapor_mole_fractions[k]` is component `k`'s local mole fraction
/// over the trace (same length as [`Self::temperature`]).
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseEnvelopeData {
    /// Temperature **\[K\]** along the trace.
    pub temperature: Vec<f64>,
    /// Pressure **\[Pa\]** along the trace.
    pub pressure: Vec<f64>,
    /// Liquid-phase molar density **\[mol/m³\]** along the trace.
    pub rhomolar_liq: Vec<f64>,
    /// Vapor-phase molar density **\[mol/m³\]** along the trace.
    pub rhomolar_vap: Vec<f64>,
    /// Liquid-phase mole fractions **\[dimensionless, from 0 to 1 each\]**, component-major
    /// (see the struct docs).
    pub liquid_mole_fractions: Vec<Vec<f64>>,
    /// Vapor-phase mole fractions **\[dimensionless, from 0 to 1 each\]**, component-major
    /// (see the struct docs).
    pub vapor_mole_fractions: Vec<Vec<f64>>,
}

impl AbstractState {
    /// Creates and returns a new [`AbstractState`] instance
    /// with specified backend and substance names.
    ///
    /// # Arguments
    ///
    /// - `backend_name` -- name of the backend _(raw [`&str`](str) or
    ///   [`Backend::name`](crate::fluid::backend::Backend::name))_
    /// - `composition_id` -- names of the substance components separated by the `&` symbol or just
    ///   a single substance name _(raw [`&str`](str) or
    ///   [`Substance::composition_id`](crate::substance::Substance::composition_id))_
    ///
    /// # Errors
    ///
    /// Returns a [`CoolPropError`] for invalid inputs.
    ///
    /// # Examples
    ///
    /// For pure substances:
    ///
    /// ```
    /// use rfluids::prelude::*;
    ///
    /// let water = AbstractState::new("HEOS", "Water");
    /// assert!(water.is_ok());
    /// ```
    ///
    /// For incompressible binary mixtures:
    ///
    /// ```
    /// use rfluids::prelude::*;
    ///
    /// let propylene_glycol = AbstractState::new("INCOMP", "MPG");
    /// assert!(propylene_glycol.is_ok());
    /// ```
    ///
    /// For mixtures:
    ///
    /// ```
    /// use rfluids::prelude::*;
    ///
    /// let mixture = AbstractState::new("HEOS", "Water&Ethanol");
    /// assert!(mixture.is_ok());
    /// ```
    ///
    /// # See Also
    ///
    /// - [CoolProp Low-Level API](https://coolprop.org/coolprop/LowLevelAPI.html)
    /// - [Pure and Pseudo-Pure Substances](https://coolprop.org/fluid_properties/PurePseudoPure.html)
    /// - [Incompressible Substances](https://coolprop.org/fluid_properties/Incompressibles.html)
    /// - [Predefined Mixtures](https://coolprop.org/coolprop/HighLevelAPI.html#predefined-mixtures)
    /// - [`Substance`](crate::substance::Substance)
    pub fn new(
        backend_name: impl AsRef<str>,
        composition_id: impl AsRef<str>,
    ) -> Result<AbstractState> {
        let backend_name = backend_name.as_ref().trim();
        let factory_exclusive = factory_requires_exclusive(backend_name);
        let state_exclusive = state_requires_exclusive(backend_name);
        let backend_name = c_string_trimmed("backend_name", backend_name)?;
        let composition_id = c_string_trimmed("composition_id", composition_id)?;
        let mut err_buffer = ErrorBuffer::default();
        let (err_code, err_message, err_buffer_capacity) = err_buffer.as_mut_parts();
        let factory = |coolprop: &bindings::CoolProp| unsafe {
            coolprop.AbstractState_factory(
                backend_name.as_ptr(),
                composition_id.as_ptr(),
                err_code,
                err_message,
                err_buffer_capacity,
            )
        };
        let handle = if factory_exclusive {
            let coolprop = COOLPROP.exclusive_access();
            factory(&coolprop)
        } else {
            let coolprop = COOLPROP.shared_access();
            factory(&coolprop)
        };
        err_buffer.into_result()?;
        Ok(Self { handle, exclusive: state_exclusive, marker: PhantomData })
    }

    /// Set the fractions _(mole, mass or volume)_[^note].
    ///
    /// [^note]:  It will be defined automatically, depending on the specified backend.
    /// For example, the `"HEOS"` backend uses mole fractions, while the `"INCOMP"` backend
    /// uses mass or volume fractions, depending on which substance is specified.
    ///
    /// # Arguments
    ///
    /// - `fractions` -- substance fractions **\[dimensionless, from 0 to 1 each\]**
    ///
    /// # Errors
    ///
    /// Returns a [`CoolPropError`] for invalid inputs.
    ///
    /// # Examples
    ///
    /// For incompressible binary mixtures:
    ///
    /// ```
    /// use rfluids::prelude::*;
    ///
    /// let mut propylene_glycol = AbstractState::new("INCOMP", "MPG")?;
    /// let res = propylene_glycol.set_fractions(&[0.6]);
    /// assert!(res.is_ok());
    /// # Ok::<(), rfluids::native::CoolPropError>(())
    /// ```
    ///
    /// For custom mixtures:
    ///
    /// ```
    /// use rfluids::prelude::*;
    ///
    /// let mut mixture = AbstractState::new("HEOS", "Water&Ethanol")?;
    /// let res = mixture.set_fractions(&[0.8, 0.2]);
    /// assert!(res.is_ok());
    /// # Ok::<(), rfluids::native::CoolPropError>(())
    /// ```
    pub fn set_fractions(&mut self, fractions: &[f64]) -> Result<()> {
        let mut err_buffer = ErrorBuffer::default();
        let (err_code, err_message, err_buffer_capacity) = err_buffer.as_mut_parts();
        self.with_coolprop(|coolprop| unsafe {
            coolprop.AbstractState_set_fractions(
                self.handle,
                fractions.as_ptr(),
                fractions.len() as c_long,
                err_code,
                err_message,
                err_buffer_capacity,
            );
        });
        err_buffer.into_result()
    }

    /// Gets the current mole fractions **\[dimensionless, from 0 to 1 each\]**,
    /// in this `AbstractState`'s fixed component order.
    ///
    /// # Arguments
    ///
    /// - `max_components` -- capacity of the read-back buffer. `CoolProp`'s C API requires this
    ///   sized up front; pass the actual expected component count if known (e.g. a
    ///   [`Substance`]'s), or [`MAX_COMPONENTS`] as a generous default otherwise.
    ///
    /// # Errors
    ///
    /// Returns a [`CoolPropError`] if `CoolProp` cannot report the composition, or reports more
    /// components than `max_components` can hold.
    ///
    /// Note that fractions never having been set isn't an error condition -- it's reported as
    /// an empty vector, not a failure (confirmed empirically; `CoolPropLib.h` doesn't document
    /// this case).
    ///
    /// # Examples
    ///
    /// ```
    /// use approx::assert_relative_eq;
    /// use rfluids::prelude::*;
    ///
    /// let mut mixture = AbstractState::new("HEOS", "Water&Ethanol")?;
    /// mixture.set_fractions(&[0.8, 0.2])?;
    /// let res = mixture.mole_fractions(2)?;
    /// assert_relative_eq!(res.as_slice(), [0.8, 0.2].as_slice());
    /// # Ok::<(), rfluids::native::CoolPropError>(())
    /// ```
    ///
    /// # See Also
    ///
    /// - [`AbstractState::mole_fractions_sat_state`]
    pub fn mole_fractions(&self, max_components: usize) -> Result<Vec<f64>> {
        let mut err_buffer = ErrorBuffer::default();
        let (err_code, err_message, err_buffer_capacity) = err_buffer.as_mut_parts();
        let mut fractions = vec![0.0_f64; max_components];
        let mut reported: c_long = 0;
        let max_components_c_long =
            c_long::try_from(max_components).expect("max_components must fit into `c_long`");
        self.with_coolprop(|coolprop| unsafe {
            coolprop.AbstractState_get_mole_fractions(
                self.handle,
                fractions.as_mut_ptr(),
                max_components_c_long,
                &raw mut reported,
                err_code,
                err_message,
                err_buffer_capacity,
            );
        });
        err_buffer.into_result()?;
        truncated_to_reported_len(fractions, reported)
    }

    /// Gets the mole fractions **\[dimensionless, from 0 to 1 each\]** of one side of the
    /// current two-phase equilibrium state, in this `AbstractState`'s fixed component order.
    ///
    /// # Arguments
    ///
    /// - `saturated_state` -- which side of the saturation dome to read _(raw [`&str`](str) or
    ///   [`SaturatedState`](crate::io::SaturatedState))_
    /// - `max_components` -- capacity of the read-back buffer. `CoolProp`'s C API requires this
    ///   sized up front; pass the actual expected component count if known (e.g. a
    ///   [`Substance`]'s), or [`MAX_COMPONENTS`] as a generous default otherwise.
    ///
    /// # Errors
    ///
    /// Returns a [`CoolPropError`] if the current state isn't two-phase, if `CoolProp` cannot
    /// otherwise report the composition, or if it reports more components than `max_components`
    /// can hold.
    ///
    /// # Examples
    ///
    /// ```
    /// use rfluids::prelude::*;
    ///
    /// let mut mixture = AbstractState::new("HEOS", "CarbonDioxide&Water")?;
    /// mixture.set_fractions(&[0.1, 0.9])?;
    /// mixture.update(FluidInputPair::PT, 2e6, 320.0)?;
    /// let liquid = mixture.mole_fractions_sat_state(SaturatedState::Liquid, 2)?;
    /// let gas = mixture.mole_fractions_sat_state(SaturatedState::Gas, 2)?;
    /// assert_eq!(liquid.len(), 2);
    /// assert_eq!(gas.len(), 2);
    /// // The gas side is enriched in `CarbonDioxide` (component `0`), the liquid side in
    /// // `Water` (component `1`).
    /// assert!(gas[0] > liquid[0]);
    /// # Ok::<(), rfluids::native::CoolPropError>(())
    /// ```
    ///
    /// # See Also
    ///
    /// - [`SaturatedState`](crate::io::SaturatedState)
    /// - [`AbstractState::mole_fractions`]
    pub fn mole_fractions_sat_state(
        &self,
        saturated_state: impl AsRef<str>,
        max_components: usize,
    ) -> Result<Vec<f64>> {
        let saturated_state = c_string_trimmed("saturated_state", saturated_state)?;
        let mut err_buffer = ErrorBuffer::default();
        let (err_code, err_message, err_buffer_capacity) = err_buffer.as_mut_parts();
        let mut fractions = vec![0.0_f64; max_components];
        let mut reported: c_long = 0;
        let max_components_c_long =
            c_long::try_from(max_components).expect("max_components must fit into `c_long`");
        self.with_coolprop(|coolprop| unsafe {
            coolprop.AbstractState_get_mole_fractions_satState(
                self.handle,
                saturated_state.as_ptr(),
                fractions.as_mut_ptr(),
                max_components_c_long,
                &raw mut reported,
                err_code,
                err_message,
                err_buffer_capacity,
            );
        });
        err_buffer.into_result()?;
        truncated_to_reported_len(fractions, reported)
    }

    /// Sets one binary interaction parameter for the `(i, j)` component pair.
    ///
    /// # Arguments
    ///
    /// - `i`, `j` -- 0-based component indices (`i < j`), in this `AbstractState`'s fixed
    ///   component order
    /// - `parameter` -- parameter name, e.g. `"betaT"`, `"gammaT"`, `"betaV"`, or `"gammaV"`
    /// - `value` -- the parameter's new value
    ///
    /// # Errors
    ///
    /// Returns a [`CoolPropError`] for an invalid index or parameter name.
    ///
    /// # Examples
    ///
    /// ```
    /// use rfluids::prelude::*;
    ///
    /// let mut mixture = AbstractState::new("HEOS", "Nitrogen&Oxygen")?;
    /// let res = mixture.set_binary_interaction_double(0, 1, "betaT", 0.999_5);
    /// assert!(res.is_ok());
    /// # Ok::<(), rfluids::native::CoolPropError>(())
    /// ```
    pub fn set_binary_interaction_double(
        &mut self,
        i: usize,
        j: usize,
        parameter: impl AsRef<str>,
        value: f64,
    ) -> Result<()> {
        let parameter = c_string_trimmed("parameter", parameter)?;
        let i = c_long::try_from(i).expect("component index must fit into `c_long`");
        let j = c_long::try_from(j).expect("component index must fit into `c_long`");
        let mut err_buffer = ErrorBuffer::default();
        let (err_code, err_message, err_buffer_capacity) = err_buffer.as_mut_parts();
        self.with_coolprop(|coolprop| unsafe {
            coolprop.AbstractState_set_binary_interaction_double(
                self.handle,
                i,
                j,
                parameter.as_ptr(),
                value,
                err_code,
                err_message,
                err_buffer_capacity,
            );
        });
        err_buffer.into_result()
    }

    /// Builds this mixture's phase envelope: the two-phase vapor-liquid-equilibrium boundary
    /// for the currently set fixed composition.
    ///
    /// Call once, after fractions and any
    /// [`set_binary_interaction_double`](Self::set_binary_interaction_double) calls, before
    /// [`phase_envelope_data`](Self::phase_envelope_data).
    ///
    /// # Errors
    ///
    /// Returns a [`CoolPropError`] if `CoolProp` can't trace it for this composition.
    ///
    /// # Examples
    ///
    /// ```
    /// use rfluids::prelude::*;
    ///
    /// let mut mixture = AbstractState::new("HEOS", "Nitrogen&Oxygen")?;
    /// mixture.set_fractions(&[0.79, 0.21])?;
    /// let res = mixture.build_phase_envelope();
    /// assert!(res.is_ok());
    /// # Ok::<(), rfluids::native::CoolPropError>(())
    /// ```
    ///
    /// # See Also
    ///
    /// - [`AbstractState::phase_envelope_data`]
    pub fn build_phase_envelope(&mut self) -> Result<()> {
        // `"level"`'s only value ever exercised against a real trace; `CoolPropLib.h` doesn't
        // document any other.
        let level = c_string_trimmed("level", "")?;
        let mut err_buffer = ErrorBuffer::default();
        let (err_code, err_message, err_buffer_capacity) = err_buffer.as_mut_parts();
        self.with_coolprop(|coolprop| unsafe {
            coolprop.AbstractState_build_phase_envelope(
                self.handle,
                level.as_ptr(),
                err_code,
                err_message,
                err_buffer_capacity,
            );
        });
        err_buffer.into_result()
    }

    /// Reads back the phase envelope built by
    /// [`build_phase_envelope`](Self::build_phase_envelope).
    ///
    /// # Arguments
    ///
    /// - `max_points` -- capacity of the per-property read-back buffer (temperature, pressure,
    ///   densities, compositions); pass a generous estimate above any trace length expected in
    ///   practice (a few hundred points is typical)
    /// - `max_components` -- capacity for the composition arrays; pass the actual expected
    ///   component count if known (e.g. a [`Substance`](crate::substance::Substance)'s)
    ///
    /// # Errors
    ///
    /// Returns a [`CoolPropError`] if `CoolProp` cannot report the trace,
    /// [`CoolPropError::PhaseEnvelopeTruncated`] if the trace filled the entire `max_points`
    /// capacity, or [`CoolPropError::TooManyComponents`] if it reports more components than
    /// `max_components` can hold.
    ///
    /// Note that calling this before [`build_phase_envelope`](Self::build_phase_envelope)
    /// isn't an error condition -- it's reported as an empty trace, not a failure (confirmed
    /// empirically; `CoolPropLib.h` doesn't document this case).
    ///
    /// # Examples
    ///
    /// ```
    /// use rfluids::prelude::*;
    ///
    /// let mut mixture = AbstractState::new("HEOS", "Nitrogen&Oxygen")?;
    /// mixture.set_fractions(&[0.79, 0.21])?;
    /// mixture.build_phase_envelope()?;
    /// let trace = mixture.phase_envelope_data(2000, 2)?;
    /// assert!(!trace.temperature.is_empty());
    /// assert_eq!(trace.liquid_mole_fractions.len(), 2);
    /// # Ok::<(), rfluids::native::CoolPropError>(())
    /// ```
    ///
    /// # See Also
    ///
    /// - [`AbstractState::build_phase_envelope`]
    /// - [`PhaseEnvelopeData`]
    pub fn phase_envelope_data(
        &mut self,
        max_points: usize,
        max_components: usize,
    ) -> Result<PhaseEnvelopeData> {
        let mut t = vec![0.0_f64; max_points];
        let mut p = vec![0.0_f64; max_points];
        let mut rhomolar_vap = vec![0.0_f64; max_points];
        let mut rhomolar_liq = vec![0.0_f64; max_points];
        let mut x = vec![0.0_f64; max_points * max_components];
        let mut y = vec![0.0_f64; max_points * max_components];
        let mut actual_length: c_long = 0;
        let mut actual_components: c_long = 0;
        let max_points_c = c_long::try_from(max_points).expect("max_points must fit into `c_long`");
        let max_components_c =
            c_long::try_from(max_components).expect("max_components must fit into `c_long`");

        let mut err_buffer = ErrorBuffer::default();
        let (err_code, err_message, err_buffer_capacity) = err_buffer.as_mut_parts();
        self.with_coolprop(|coolprop| unsafe {
            coolprop.AbstractState_get_phase_envelope_data_checkedMemory(
                self.handle,
                max_points_c,
                max_components_c,
                t.as_mut_ptr(),
                p.as_mut_ptr(),
                rhomolar_vap.as_mut_ptr(),
                rhomolar_liq.as_mut_ptr(),
                x.as_mut_ptr(),
                y.as_mut_ptr(),
                &raw mut actual_length,
                &raw mut actual_components,
                err_code,
                err_message,
                err_buffer_capacity,
            );
        });
        err_buffer.into_result()?;

        let reported_points = usize::try_from(actual_length).unwrap_or(0);
        if reported_points >= max_points {
            return Err(CoolPropError::PhaseEnvelopeTruncated { capacity: max_points });
        }
        let reported_components = usize::try_from(actual_components).unwrap_or(0);
        if reported_components > max_components {
            return Err(CoolPropError::TooManyComponents {
                reported: reported_components,
                capacity: max_components,
            });
        }

        t.truncate(reported_points);
        p.truncate(reported_points);
        rhomolar_vap.truncate(reported_points);
        rhomolar_liq.truncate(reported_points);

        // Component-major, stride = `reported_points` (the true, dynamic trace length `CoolProp`
        // discovered while tracing) -- *not* `max_points`, and not point-major. Confirmed
        // empirically (a 3-component HEOS mixture, checking where per-point mole fractions sum
        // to 1): getting this wrong (component-major-by-`max_points`, or point-major) silently
        // produces plausible-looking but wrong composition data.
        let liquid_mole_fractions: Vec<Vec<f64>> = (0..reported_components)
            .map(|k| x[k * reported_points..(k + 1) * reported_points].to_vec())
            .collect();
        let vapor_mole_fractions: Vec<Vec<f64>> = (0..reported_components)
            .map(|k| y[k * reported_points..(k + 1) * reported_points].to_vec())
            .collect();

        Ok(PhaseEnvelopeData {
            temperature: t,
            pressure: p,
            rhomolar_liq,
            rhomolar_vap,
            liquid_mole_fractions,
            vapor_mole_fractions,
        })
    }

    /// Update the state of the fluid.
    ///
    /// # Arguments
    ///
    /// - `input_pair_key` -- input pair key _(raw [`u8`] or
    ///   [`FluidInputPair`](crate::io::FluidInputPair))_
    /// - `input1` -- value of the first input property **\[SI units\]**
    /// - `input2` -- value of the second input property **\[SI units\]**
    ///
    /// # Errors
    ///
    /// Returns a [`CoolPropError`] for invalid inputs.
    ///
    /// # Examples
    ///
    /// ```
    /// use rfluids::prelude::*;
    ///
    /// let mut water = AbstractState::new("HEOS", "Water")?;
    /// let res = water.update(FluidInputPair::PT, 101_325.0, 293.15);
    /// assert!(res.is_ok());
    /// # Ok::<(), rfluids::native::CoolPropError>(())
    /// ```
    ///
    /// # See Also
    ///
    /// - [`FluidInputPair`](crate::io::FluidInputPair)
    pub fn update(
        &mut self,
        input_pair_key: impl Into<u8>,
        input1: f64,
        input2: f64,
    ) -> Result<()> {
        let mut err_buffer = ErrorBuffer::default();
        let (err_code, err_message, err_buffer_capacity) = err_buffer.as_mut_parts();
        self.with_coolprop(|coolprop| unsafe {
            coolprop.AbstractState_update(
                self.handle,
                c_long::from(input_pair_key.into()),
                input1,
                input2,
                err_code,
                err_message,
                err_buffer_capacity,
            );
        });
        err_buffer.into_result()
    }

    /// Returns an output parameter value **\[SI units\]**
    ///
    /// # Arguments
    ///
    /// - `key` -- output parameter key _(raw [`u8`], [`FluidParam`](crate::io::FluidParam) or
    ///   [`FluidTrivialParam`](crate::io::FluidTrivialParam))_
    ///
    /// # Errors
    ///
    /// Returns a [`CoolPropError`] for non-trivial outputs with undefined state or invalid inputs.
    ///
    /// # Examples
    ///
    /// ## Pure substances
    ///
    /// To calculate the specific heat **\[J/kg/K\]** of saturated water vapor at _1 atm_:
    ///
    /// ```
    /// use approx::assert_relative_eq;
    /// use rfluids::prelude::*;
    ///
    /// let mut water = AbstractState::new("HEOS", "Water")?;
    /// water.update(FluidInputPair::PQ, 101_325.0, 1.0)?;
    /// let res = water.keyed_output(FluidParam::CpMass)?;
    /// assert_relative_eq!(res, 2_079.937_085_633_241, max_relative = 1e-6);
    /// # Ok::<(), rfluids::native::CoolPropError>(())
    /// ```
    ///
    /// ## Incompressible binary mixtures
    ///
    /// To calculate the dynamic viscosity **\[Pa·s\]** of propylene glycol aqueous solution
    /// with _60 %_ mass fraction at _100 kPa_ and _-20 °C_:
    ///
    /// ```
    /// use approx::assert_relative_eq;
    /// use rfluids::prelude::*;
    ///
    /// let mut propylene_glycol = AbstractState::new("INCOMP", "MPG")?;
    /// propylene_glycol.set_fractions(&[0.6])?;
    /// propylene_glycol.update(FluidInputPair::PT, 100e3, 253.15)?;
    /// let res = propylene_glycol.keyed_output(FluidParam::DynamicViscosity)?;
    /// assert_relative_eq!(res, 0.139_073_910_539_388_47, max_relative = 1e-6);
    /// # Ok::<(), rfluids::native::CoolPropError>(())
    /// ```
    ///
    /// ## Custom mixtures
    ///
    /// To calculate the density **\[kg/m³\]** of ethanol aqueous solution
    /// (with ethanol _20 %_ mole fraction) at _200 kPa_ and _4 °C_:
    ///
    /// ```
    /// use approx::assert_relative_eq;
    /// use rfluids::prelude::*;
    ///
    /// let mut mixture = AbstractState::new("HEOS", "Water&Ethanol")?;
    /// mixture.set_fractions(&[0.8, 0.2])?;
    /// mixture.update(FluidInputPair::PT, 200e3, 277.15)?;
    /// let res = mixture.keyed_output(FluidParam::DMass)?;
    /// assert_relative_eq!(res, 944.757_029_911_081_5, max_relative = 1e-6);
    /// # Ok::<(), rfluids::native::CoolPropError>(())
    /// ```
    ///
    /// # See Also
    ///
    /// - [`FluidParam`](crate::io::FluidParam)
    /// - [`FluidTrivialParam`](crate::io::FluidTrivialParam)
    pub fn keyed_output(&self, key: impl Into<u8>) -> Result<f64> {
        let mut err_buffer = ErrorBuffer::default();
        let (err_code, err_message, err_buffer_capacity) = err_buffer.as_mut_parts();
        let key = key.into();
        let value = self.with_coolprop(|coolprop| unsafe {
            coolprop.AbstractState_keyed_output(
                self.handle,
                c_long::from(key),
                err_code,
                err_message,
                err_buffer_capacity,
            )
        });
        err_buffer.into_result()?;
        if !value.is_finite() {
            return Err(CoolPropError::NonFiniteKeyedOutput { key });
        }
        Ok(value)
    }

    /// Specify the phase state for all further calculations.
    ///
    /// # Arguments
    ///
    /// - `phase` -- phase state _(raw [`&str`](str) or [`Phase`](crate::io::Phase))_
    ///
    /// # Errors
    ///
    /// Returns a [`CoolPropError`] for invalid inputs.
    ///
    /// # Examples
    ///
    /// ```
    /// use rfluids::prelude::*;
    ///
    /// let mut water = AbstractState::new("HEOS", "Water")?;
    /// water.specify_phase(Phase::Liquid)?;
    /// let mut res = water.update(FluidInputPair::PT, 101_325.0, 293.15);
    /// assert!(res.is_ok());
    /// water.specify_phase(Phase::Gas)?;
    /// res = water.update(FluidInputPair::PT, 101_325.0, 293.15);
    /// assert!(res.is_err());
    /// # Ok::<(), rfluids::native::CoolPropError>(())
    /// ```
    ///
    /// # See Also
    ///
    /// - [Imposing the Phase (Optional)](https://coolprop.org/coolprop/HighLevelAPI.html#imposing-the-phase-optional)
    /// - [`Phase`](crate::io::Phase)
    pub fn specify_phase(&mut self, phase: impl AsRef<str>) -> Result<()> {
        let phase = c_string_trimmed("phase", phase)?;
        let mut err_buffer = ErrorBuffer::default();
        let (err_code, err_message, err_buffer_capacity) = err_buffer.as_mut_parts();
        self.with_coolprop(|coolprop| unsafe {
            coolprop.AbstractState_specify_phase(
                self.handle,
                phase.as_ptr(),
                err_code,
                err_message,
                err_buffer_capacity,
            );
        });
        err_buffer.into_result()
    }

    /// Unspecify the phase state and go back to calculating it based on the inputs.
    ///
    /// # Examples
    ///
    /// ```
    /// use rfluids::prelude::*;
    ///
    /// let mut water = AbstractState::new("HEOS", "Water")?;
    /// water.specify_phase(Phase::Gas)?;
    /// let mut res = water.update(FluidInputPair::PT, 101_325.0, 293.15);
    /// assert!(res.is_err());
    /// water.unspecify_phase();
    /// res = water.update(FluidInputPair::PT, 101_325.0, 293.15);
    /// assert!(res.is_ok());
    /// # Ok::<(), rfluids::native::CoolPropError>(())
    /// ```
    ///
    /// # See Also
    ///
    /// - [Imposing the Phase (Optional)](https://coolprop.org/coolprop/HighLevelAPI.html#imposing-the-phase-optional)
    pub fn unspecify_phase(&mut self) {
        let mut err_buffer = ErrorBuffer::default();
        let (err_code, err_message, err_buffer_capacity) = err_buffer.as_mut_parts();
        self.with_coolprop(|coolprop| unsafe {
            coolprop.AbstractState_unspecify_phase(
                self.handle,
                err_code,
                err_message,
                err_buffer_capacity,
            );
        });
    }

    fn with_coolprop<T>(&self, call: impl FnOnce(&bindings::CoolProp) -> T) -> T {
        if self.exclusive {
            let coolprop = COOLPROP.exclusive_access();
            call(&coolprop)
        } else {
            let coolprop = COOLPROP.shared_access();
            call(&coolprop)
        }
    }
}

/// Truncates a `mole_fractions`/`mole_fractions_sat_state` read-back buffer to `CoolProp`'s
/// reported component count, or fails if that count exceeds the buffer's capacity.
fn truncated_to_reported_len(mut fractions: Vec<f64>, reported: c_long) -> Result<Vec<f64>> {
    let capacity = fractions.len();
    let reported = usize::try_from(reported).unwrap_or(0);
    if reported > capacity {
        return Err(CoolPropError::TooManyComponents { reported, capacity });
    }
    fractions.truncate(reported);
    Ok(fractions)
}

impl TryFrom<&SubstanceWithBackend> for AbstractState {
    type Error = CoolPropError;

    fn try_from(value: &SubstanceWithBackend) -> Result<Self> {
        let (component_names, fractions): (Cow<'static, str>, Option<Vec<f64>>) =
            match &value.substance {
                Substance::Pure(pure) => (Cow::Borrowed(pure.into()), None),
                Substance::IncompPure(incomp_pure) => (Cow::Borrowed(incomp_pure.into()), None),
                Substance::PredefinedMix(predefined_mix) => {
                    (Cow::Borrowed(predefined_mix.into()), None)
                }
                Substance::BinaryMix(binary_mix) => {
                    (Cow::Borrowed(binary_mix.kind.into()), Some(vec![binary_mix.fraction]))
                }
                Substance::CustomMix(custom_mix) => {
                    let mix = custom_mix.clone().into_mole_based();
                    // Sorted by name (not fraction) so this order stays fixed regardless of
                    // composition -- required since `set_fractions` only overwrites values at
                    // fixed positions, never reorders them.
                    let (components, fractions): (Vec<&str>, Vec<f64>) = mix
                        .sorted_by_name()
                        .into_iter()
                        .map(|(pure, fraction)| (<&str>::from(pure), fraction))
                        .unzip();
                    (Cow::Owned(components.join("&")), Some(fractions))
                }
            };
        let mut backend = AbstractState::new(value.backend.name(), component_names)?;
        match fractions {
            Some(fractions) => backend.set_fractions(&fractions).map(|()| backend),
            None => Ok(backend),
        }
    }
}

impl Drop for AbstractState {
    fn drop(&mut self) {
        let mut err_buffer = ErrorBuffer::default();
        let (err_code, err_message, err_buffer_capacity) = err_buffer.as_mut_parts();
        self.with_coolprop(|coolprop| unsafe {
            coolprop.AbstractState_free(self.handle, err_code, err_message, err_buffer_capacity);
        });
        // `Drop` cannot report a cleanup error. A valid, uniquely owned handle is freed exactly
        // once; any error here indicates an invariant violation or external raw-API interference.
        let _cleanup_result = err_buffer.into_result();
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Barrier, thread};

    use rayon::prelude::*;
    use rstest::*;
    use static_assertions::{assert_impl_all, assert_not_impl_any};

    use super::*;
    use crate::{
        io::{FluidInputPair, FluidParam, Phase, SaturatedState},
        test::assert_relative_eq,
    };

    assert_impl_all!(AbstractState: Send);
    assert_not_impl_any!(AbstractState: Sync, Clone);

    #[derive(Clone, Copy)]
    struct SharedBackendCase {
        backend: &'static str,
        substance: &'static str,
        fraction: Option<f64>,
        input_pair: FluidInputPair,
        input1: f64,
        input2: f64,
    }

    #[test]
    fn thread_safety() {
        // Given
        let backend = "HEOS";
        let substance = "Water";
        let pressure_range = 101_000..101_500;
        let quality = 0.0;

        // When
        let res: Vec<Result<f64>> = pressure_range
            .into_par_iter()
            .map(move |p| {
                let mut sut = AbstractState::new(backend, substance).unwrap();
                sut.specify_phase(Phase::TwoPhase).unwrap();
                sut.update(FluidInputPair::PQ, p.into(), quality).unwrap();
                sut.keyed_output(FluidParam::T)
            })
            .collect();

        // Then
        assert!(res.iter().all(Result::is_ok));
    }

    #[test]
    fn shared_backends_run_independent_states_in_parallel() {
        // Given
        const CASES: [SharedBackendCase; 6] = [
            SharedBackendCase {
                backend: "HEOS",
                substance: "Water",
                fraction: None,
                input_pair: FluidInputPair::PT,
                input1: 101_325.0,
                input2: 300.0,
            },
            SharedBackendCase {
                backend: "INCOMP",
                substance: "MPG",
                fraction: Some(0.4),
                input_pair: FluidInputPair::PT,
                input1: 101_325.0,
                input2: 300.0,
            },
            SharedBackendCase {
                backend: "IF97",
                substance: "Water",
                fraction: None,
                input_pair: FluidInputPair::PT,
                input1: 101_325.0,
                input2: 300.0,
            },
            SharedBackendCase {
                backend: "SRK",
                substance: "Propane",
                fraction: None,
                input_pair: FluidInputPair::PT,
                input1: 101_325.0,
                input2: 300.0,
            },
            SharedBackendCase {
                backend: "PR",
                substance: "Propane",
                fraction: None,
                input_pair: FluidInputPair::PT,
                input1: 101_325.0,
                input2: 300.0,
            },
            SharedBackendCase {
                backend: "PCSAFT",
                substance: "METHANE",
                fraction: None,
                input_pair: FluidInputPair::DMolarT,
                input1: 40.0,
                input2: 300.0,
            },
        ];
        const CALLS_PER_THREAD: usize = 16;

        fn density(case: SharedBackendCase) -> f64 {
            let mut state = AbstractState::new(case.backend, case.substance)
                .expect("shared backend state should be created");
            if let Some(fraction) = case.fraction {
                state.set_fractions(&[fraction]).expect("fractions should be set");
            }
            state
                .update(case.input_pair, case.input1, case.input2)
                .expect("shared backend state should be updated");
            state.keyed_output(FluidParam::DMass).expect("density should be calculated")
        }

        let expected = CASES.iter().copied().map(density).collect::<Vec<_>>();
        let barrier = Barrier::new(CASES.len());

        // When
        let actual = thread::scope(|scope| {
            let threads = CASES
                .into_iter()
                .enumerate()
                .map(|(case_index, case)| {
                    let barrier = &barrier;
                    scope.spawn(move || {
                        barrier.wait();
                        let values =
                            (0..CALLS_PER_THREAD).map(|_| density(case)).collect::<Vec<_>>();
                        (case_index, values)
                    })
                })
                .collect::<Vec<_>>();
            threads
                .into_iter()
                .map(|thread| thread.join().expect("calculation thread should not panic"))
                .collect::<Vec<_>>()
        });

        // Then
        for (case_index, values) in actual {
            for actual in values {
                assert_relative_eq!(actual, expected[case_index]);
            }
        }
    }

    #[rstest]
    #[case("HEOS", false, false)]
    #[case(" heos ", false, false)]
    #[case("INCOMP", false, false)]
    #[case("IF97", false, false)]
    #[case("SRK", false, false)]
    #[case("PR", false, false)]
    #[case("PCSAFT", false, false)]
    #[case("VTPR", true, false)]
    #[case(" vtpr ", true, false)]
    #[case("REFPROP", true, true)]
    #[case("TTSE&HEOS", true, true)]
    #[case("BICUBIC&HEOS", true, true)]
    #[case("SVDSBTL&HEOS", true, false)]
    #[case(" svdsbtl&if97 ", true, false)]
    #[case("SVDSBTL&REFPROP", true, true)]
    #[case("UNKNOWN", true, true)]
    fn backend_access_classification(
        #[case] backend: &str,
        #[case] expected_factory_exclusive: bool,
        #[case] expected_state_exclusive: bool,
    ) {
        // Given
        let expected = (expected_factory_exclusive, expected_state_exclusive);

        // When
        let actual = (factory_requires_exclusive(backend), state_requires_exclusive(backend));

        // Then
        assert_eq!(actual, expected);
    }

    #[rstest]
    #[case("HEOS", "Water")]
    #[case("INCOMP", "MPG")]
    #[case("HEOS", "Water&Ethanol")]
    fn new_valid_inputs(#[case] backend_name: &str, #[case] substance_names: &str) {
        // When
        let res = AbstractState::new(backend_name, substance_names);

        // Then
        assert!(res.is_ok());
    }

    #[rstest]
    #[case(
        "Hello, World!",
        "Water",
        "Error: Invalid backend name [Hello, World!] to factory function"
    )]
    #[case(
        "INCOMP",
        "Hello, World!",
        "Error: key [Hello, World!] was not found in string_to_index_map \
        in JSONIncompressibleLibrary"
    )]
    #[case(
        "HEOS",
        "Water+Ethanol",
        "Error: key [Water+Ethanol] was not found in string_to_index_map \
        in JSONFluidLibrary"
    )]
    fn new_invalid_inputs(
        #[case] backend_name: &str,
        #[case] substance_names: &str,
        #[case] expected_message: &str,
    ) {
        // When
        let res = AbstractState::new(backend_name, substance_names).unwrap_err();

        // Then
        assert_eq!(res, CoolPropError::Native(expected_message.into()));
    }

    #[test]
    fn new_error_without_native_message() {
        // Given
        const OVERSIZED_ERROR_MARKER_LENGTH: usize = 1_000;
        let backend_name = "X".repeat(OVERSIZED_ERROR_MARKER_LENGTH);

        // When
        let res = AbstractState::new(backend_name.as_str(), "Water").unwrap_err();

        // Then
        assert_eq!(
            res,
            CoolPropError::Native(
                "CoolProp native call failed with error code 2 and no error message".into()
            )
        );
    }

    #[test]
    fn new_interior_nul_backend_name() {
        // When
        let res = AbstractState::new("HEOS\0", "Water").unwrap_err();

        // Then
        assert_eq!(res, CoolPropError::InteriorNul { arg: "backend_name", pos: 4 });
    }

    #[test]
    fn new_interior_nul_composition_id() {
        // When
        let res = AbstractState::new("HEOS", "Water\0").unwrap_err();

        // Then
        assert_eq!(res, CoolPropError::InteriorNul { arg: "composition_id", pos: 5 });
    }

    #[test]
    fn set_fractions_valid_input() {
        // Given
        let mut sut = AbstractState::new("HEOS", "Water&Ethanol").unwrap();

        // When
        let res = sut.set_fractions(&[0.6, 0.4]);

        // Then
        assert!(res.is_ok());
    }

    #[test]
    fn set_fractions_invalid_input() {
        // Given
        let mut sut = AbstractState::new("HEOS", "Water&Ethanol").unwrap();

        // When
        let res = sut.set_fractions(&[0.6, 0.4, 0.6]).unwrap_err();

        // Then
        assert_eq!(
            res,
            CoolPropError::Native(
                "Error: size of mole fraction vector [3] \
                does not equal that of component vector [2]"
                    .into()
            )
        );
    }

    #[test]
    fn mole_fractions_round_trips_set_fractions() {
        // Given
        let mut sut = AbstractState::new("HEOS", "Water&Ethanol").unwrap();
        sut.set_fractions(&[0.8, 0.2]).unwrap();

        // When
        let res = sut.mole_fractions(2).unwrap();

        // Then
        assert_relative_eq!(res.as_slice(), [0.8, 0.2].as_slice());
    }

    #[test]
    fn mole_fractions_not_yet_set() {
        // Given
        let sut = AbstractState::new("HEOS", "Water&Ethanol").unwrap();

        // When
        let res = sut.mole_fractions(MAX_COMPONENTS);

        // Then
        assert_eq!(res, Ok(Vec::new()));
    }

    /// `CoolProp` itself already refuses to write past `maxN` (raising a native error) rather
    /// than reporting a true count larger than the buffer -- confirmed empirically, since
    /// `CoolPropLib.h` doesn't document this. This means [`CoolPropError::TooManyComponents`] is
    /// unreachable through this call in practice; it stays as defense-in-depth against a
    /// backend that might one day behave differently, rather than silently mis-truncating.
    #[test]
    fn mole_fractions_max_components_too_small() {
        // Given
        let mut sut = AbstractState::new("HEOS", "Water&Ethanol").unwrap();
        sut.set_fractions(&[0.8, 0.2]).unwrap();

        // When
        let res = sut.mole_fractions(1);

        // Then
        assert!(matches!(res, Err(CoolPropError::Native(_))));
    }

    /// Pins down `AbstractState_get_mole_fractions_satState`'s undocumented
    /// `saturated_state` argument against a known two-phase CO2-Water point:
    /// `"liquid"` and `"gas"` are both accepted and return the physically-expected split, while
    /// `"vapor"` -- a plausible-looking spelling for the gas side -- is rejected.
    #[test]
    fn mole_fractions_sat_state_liquid_and_gas_split_a_known_two_phase_point() {
        // Given
        let mut sut = AbstractState::new("HEOS", "CarbonDioxide&Water").unwrap();
        sut.set_fractions(&[0.1, 0.9]).unwrap();
        sut.update(FluidInputPair::PT, 2.0e6, 320.0).unwrap();

        // When
        let liquid = sut.mole_fractions_sat_state("liquid", 2).unwrap();
        let gas = sut.mole_fractions_sat_state("gas", 2).unwrap();
        let vapor = sut.mole_fractions_sat_state("vapor", 2);

        // Then
        assert_eq!(liquid.len(), 2);
        assert_eq!(gas.len(), 2);
        assert_relative_eq!(liquid.iter().sum::<f64>(), 1.0);
        assert_relative_eq!(gas.iter().sum::<f64>(), 1.0);
        // CO2 (component 0) is enriched in the gas side, water (component 1) in the liquid side.
        assert!(gas[0] > liquid[0]);
        assert!(liquid[1] > gas[1]);
        assert!(vapor.is_err());
    }

    #[test]
    fn mole_fractions_sat_state_single_phase_state() {
        // Given
        let mut sut = AbstractState::new("HEOS", "CarbonDioxide&Water").unwrap();
        sut.set_fractions(&[0.1, 0.9]).unwrap();
        sut.update(FluidInputPair::PT, 1e4, 500.0).unwrap();

        // When
        let res = sut.mole_fractions_sat_state(SaturatedState::Liquid, 2);

        // Then
        assert!(res.is_err());
    }

    #[test]
    fn set_binary_interaction_double_valid_input() {
        // Given
        let mut sut = AbstractState::new("HEOS", "Nitrogen&Oxygen").unwrap();

        // When
        let res = sut.set_binary_interaction_double(0, 1, "betaT", 0.999_5);

        // Then
        assert!(res.is_ok());
    }

    #[test]
    fn set_binary_interaction_double_invalid_parameter() {
        // Given
        let mut sut = AbstractState::new("HEOS", "Nitrogen&Oxygen").unwrap();

        // When
        let res = sut.set_binary_interaction_double(0, 1, "not_a_real_parameter", 1.0);

        // Then
        assert!(res.is_err());
    }

    /// Pins down `AbstractState_get_phase_envelope_data_checkedMemory`'s undocumented
    /// composition-array layout: component-major, stride = the true trace length, not `capacity`
    /// and not point-major -- confirmed by checking every point's liquid/vapor mole fractions
    /// sum to 1.
    #[test]
    fn phase_envelope_data_traces_a_known_binary_pair() {
        // Given
        let mut sut = AbstractState::new("HEOS", "Nitrogen&Oxygen").unwrap();
        sut.set_fractions(&[0.79, 0.21]).unwrap();
        sut.build_phase_envelope().unwrap();

        // When
        let trace = sut.phase_envelope_data(2000, 2).unwrap();

        // Then
        // NB: composition values themselves aren't checked here -- `CoolProp`'s raw x/y arrays
        // have a confirmed quirk for exactly 2 components (`y` just echoes the feed fraction
        // back, `x` interleaves two families of values point by point), worked around one layer
        // up in `phase_envelope::binary_mixture`, not here. See
        // `phase_envelope_data_ternary_mole_fractions_sum_to_one` below for a clean (N >= 3,
        // quirk-free) validation of the stride/layout logic itself.
        assert!(!trace.temperature.is_empty());
        assert_eq!(trace.pressure.len(), trace.temperature.len());
        assert_eq!(trace.rhomolar_liq.len(), trace.temperature.len());
        assert_eq!(trace.rhomolar_vap.len(), trace.temperature.len());
        assert_eq!(trace.liquid_mole_fractions.len(), 2);
        assert_eq!(trace.vapor_mole_fractions.len(), 2);
        for component in &trace.liquid_mole_fractions {
            assert_eq!(component.len(), trace.temperature.len());
        }
    }

    /// A ternary mixture, checking the composition-array bookkeeping (component count, per-
    /// component length) rather than composition *values* -- those carry the same quirk
    /// documented for exactly 2 components (confirmed empirically to extend to 3: `y` cycles
    /// through permutations of the feed fractions rather than tracing a smooth per-component
    /// curve). Interpreting/de-interleaving that belongs one layer up, in
    /// `phase_envelope::binary_mixture`, which already does it for the 2-component case and is
    /// validated end-to-end by `tests/binary_mixture.rs`; this layer's job is only to marshal
    /// `CoolProp`'s raw arrays faithfully.
    #[test]
    fn phase_envelope_data_ternary_mixture() {
        // Given
        let mut sut = AbstractState::new("HEOS", "Nitrogen&Oxygen&Argon").unwrap();
        sut.set_fractions(&[0.78, 0.21, 0.01]).unwrap();
        sut.build_phase_envelope().unwrap();

        // When
        let trace = sut.phase_envelope_data(2000, 3).unwrap();

        // Then
        assert!(!trace.temperature.is_empty());
        assert_eq!(trace.liquid_mole_fractions.len(), 3);
        assert_eq!(trace.vapor_mole_fractions.len(), 3);
        for component in trace.liquid_mole_fractions.iter().chain(&trace.vapor_mole_fractions) {
            assert_eq!(component.len(), trace.temperature.len());
        }
    }

    /// Not calling `build_phase_envelope` first isn't an error condition -- it's reported as an
    /// empty trace, not a failure (confirmed empirically; `CoolPropLib.h` doesn't document this
    /// case), matching [`AbstractState::mole_fractions`]'s analogous
    /// nothing-set-yet-isn't-an-error behavior.
    #[test]
    fn phase_envelope_data_without_building_first() {
        // Given
        let mut sut = AbstractState::new("HEOS", "Nitrogen&Oxygen").unwrap();
        sut.set_fractions(&[0.79, 0.21]).unwrap();

        // When
        let res = sut.phase_envelope_data(2000, 2).unwrap();

        // Then
        assert!(res.temperature.is_empty());
    }

    /// `CoolProp` itself already refuses to write past `max_points` (raising a native error)
    /// rather than reporting a true length larger than the buffer -- confirmed empirically,
    /// same finding as `mole_fractions_max_components_too_small`. This means
    /// [`CoolPropError::PhaseEnvelopeTruncated`] is unreachable through this call in practice;
    /// it stays as defense-in-depth.
    #[test]
    fn phase_envelope_data_max_points_too_small() {
        // Given
        let mut sut = AbstractState::new("HEOS", "Nitrogen&Oxygen").unwrap();
        sut.set_fractions(&[0.79, 0.21]).unwrap();
        sut.build_phase_envelope().unwrap();

        // When
        let res = sut.phase_envelope_data(1, 2);

        // Then
        assert!(matches!(res, Err(CoolPropError::Native(_))));
    }

    #[test]
    fn update_valid_inputs() {
        // Given
        let mut sut = AbstractState::new("HEOS", "Water").unwrap();

        // When
        let res = sut.update(FluidInputPair::PT, 101_325.0, 293.15);

        // Then
        assert!(res.is_ok());
    }

    #[test]
    fn update_invalid_inputs() {
        // Given
        let mut sut = AbstractState::new("HEOS", "Water").unwrap();

        // When
        let res = sut.update(FluidInputPair::PQ, 101_325.0, -1.0).unwrap_err();

        // Then
        assert_eq!(
            res,
            CoolPropError::Native("Error: Input vapor quality [Q] must be between 0 and 1".into())
        );
    }

    #[test]
    fn keyed_output_valid_state() {
        // Given
        let mut sut = AbstractState::new("HEOS", "Water").unwrap();
        sut.update(FluidInputPair::PQ, 101_325.0, 1.0).unwrap();

        // When
        let res = sut.keyed_output(FluidParam::CpMass).unwrap();

        // Then
        assert_relative_eq!(res, 2_079.937_085_633_241);
    }

    #[test]
    fn keyed_output_invalid_input() {
        // Given
        let sut = AbstractState::new("HEOS", "Water").unwrap();

        // When
        let res = sut.keyed_output(255).unwrap_err();

        // Then
        assert_eq!(
            res,
            CoolPropError::Native(
                "Error: Unable to match the key [255] in get_parameter_information for info [short]"
                    .into()
            )
        );
    }

    #[test]
    fn keyed_output_non_trivial_with_not_defined_state() {
        // Given
        let sut = AbstractState::new("HEOS", "Water").unwrap();

        // When
        let res = sut.keyed_output(FluidParam::DMass).unwrap_err();

        // Then
        assert_eq!(res, CoolPropError::NonFiniteKeyedOutput { key: 40 });
    }

    #[test]
    fn specify_phase_valid() {
        // Given
        let mut sut = AbstractState::new("HEOS", "Water").unwrap();

        // When
        sut.specify_phase(Phase::Liquid).unwrap();
        let res1 = sut.update(FluidInputPair::PT, 101_325.0, 293.15);
        sut.specify_phase(Phase::Gas).unwrap();
        let res2 = sut.update(FluidInputPair::PT, 101_325.0, 293.15);

        // Then
        assert!(res1.is_ok());
        assert!(res2.is_err());
    }

    #[test]
    fn specify_phase_invalid() {
        // Given
        let mut sut = AbstractState::new("HEOS", "Water").unwrap();

        // When
        let res = sut.specify_phase("Hello, World!").unwrap_err();

        // Then
        assert_eq!(
            res,
            CoolPropError::Native(
                "Error: Your input name [Hello, World!] is not valid \
                in get_phase_index (names are case sensitive)"
                    .into()
            )
        );
    }

    #[test]
    fn specify_phase_interior_nul() {
        // Given
        let mut sut = AbstractState::new("HEOS", "Water").unwrap();

        // When
        let res = sut.specify_phase("phase_liquid\0").unwrap_err();

        // Then
        assert_eq!(res, CoolPropError::InteriorNul { arg: "phase", pos: 12 });
    }

    #[test]
    fn unspecify_phase() {
        // Given
        let mut sut = AbstractState::new("HEOS", "Water").unwrap();

        // When
        sut.specify_phase(Phase::Gas).unwrap();
        let res1 = sut.update(FluidInputPair::PT, 101_325.0, 293.15);
        sut.unspecify_phase();
        let res2 = sut.update(FluidInputPair::PT, 101_325.0, 293.15);

        // Then
        assert!(res1.is_err());
        assert!(res2.is_ok());
    }

    #[test]
    fn unspecify_phase_for_exclusive_state() {
        // Given
        let mut sut = AbstractState::new("TTSE&HEOS", "Water").unwrap();
        sut.specify_phase(Phase::Gas).unwrap();

        // When
        sut.unspecify_phase();
        let res = sut.update(FluidInputPair::PT, 101_325.0, 293.15);

        // Then
        assert!(res.is_ok());
    }

    #[test]
    fn drop_releases_native_handle() {
        // Given
        let sut = AbstractState::new("HEOS", "Water").unwrap();
        let handle = sut.handle;

        // When
        drop(sut);
        let mut err_buffer = ErrorBuffer::default();
        let (err_code, err_message, err_buffer_capacity) = err_buffer.as_mut_parts();
        {
            let coolprop = COOLPROP.shared_access();
            unsafe {
                coolprop.AbstractState_free(handle, err_code, err_message, err_buffer_capacity);
            }
        }
        let res = err_buffer.into_result().unwrap_err();

        // Then
        assert_eq!(res, CoolPropError::Native("HandleError: could not free handle".into()));
    }
}
