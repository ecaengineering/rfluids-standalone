use std::collections::HashMap;

use super::{
    Fluid, FluidOutputError, FluidPhaseError, FluidStateError, OutputResult, StateResult,
    backend::Backend,
    common::{cached_output, guard},
    request::FluidUpdateRequest,
};
use crate::{
    io::{FluidInput, FluidTrivialParam, Phase},
    native::PhaseEnvelopeData,
    ops::mul,
    state_variant::StateVariant,
    substance::{BinaryMix, CustomMix, Pure, Substance},
};

impl<S: StateVariant> Fluid<S> {
    /// Specified substance.
    #[must_use]
    pub fn substance(&self) -> &Substance {
        &self.substance
    }

    /// Specified `CoolProp` backend.
    #[must_use]
    pub fn backend(&self) -> Backend {
        self.backend_variant
    }

    /// Currently imposed phase used as a hint for future state updates.
    ///
    /// This value reflects phase configuration, not the calculated phase of the current state.
    /// Use [`Fluid::phase`](crate::fluid::Fluid::phase) to retrieve the calculated phase.
    #[must_use]
    pub fn specified_phase(&self) -> Phase {
        self.specified_phase
    }

    /// Sets one binary interaction parameter for the `(i, j)` component pair, and returns a
    /// mutable reference to itself.
    ///
    /// # Arguments
    ///
    /// - `i`, `j` -- 0-based component indices (`i < j`), in this fluid's fixed component order
    /// - `parameter` -- parameter name, e.g. `"betaT"`, `"gammaT"`, `"betaV"`, or `"gammaV"`
    /// - `value` -- the parameter's new value
    ///
    /// # Errors
    ///
    /// Returns a [`FluidStateError::UpdateFailed`] for an invalid index or parameter name.
    ///
    /// # Examples
    ///
    /// ```
    /// use rfluids::prelude::*;
    ///
    /// let mut mixture: Fluid<Undefined> =
    ///     CustomMix::mole_based([(Pure::Nitrogen, 0.79), (Pure::Oxygen, 0.21)])?.try_into()?;
    /// let res = mixture.set_binary_interaction_double(0, 1, "betaT", 0.999_5);
    /// assert!(res.is_ok());
    /// # Ok::<(), rfluids::Error>(())
    /// ```
    pub fn set_binary_interaction_double(
        &mut self,
        i: usize,
        j: usize,
        parameter: impl AsRef<str>,
        value: f64,
    ) -> StateResult<&mut Self> {
        self.backend.set_binary_interaction_double(i, j, parameter, value)?;
        self.outputs.clear();
        self.trivial_outputs.clear();
        Ok(self)
    }

    /// Builds this fluid's phase envelope: the two-phase vapor-liquid-equilibrium boundary for
    /// the currently set fixed composition. Returns a mutable reference to itself.
    ///
    /// Call once, after fractions and any
    /// [`set_binary_interaction_double`](Self::set_binary_interaction_double) calls, before
    /// [`phase_envelope_data`](Self::phase_envelope_data).
    ///
    /// # Errors
    ///
    /// Returns a [`FluidStateError::UpdateFailed`] if `CoolProp` can't trace it for this
    /// composition.
    ///
    /// # Examples
    ///
    /// ```
    /// use rfluids::prelude::*;
    ///
    /// let mut mixture: Fluid<Undefined> =
    ///     CustomMix::mole_based([(Pure::Nitrogen, 0.79), (Pure::Oxygen, 0.21)])?.try_into()?;
    /// let res = mixture.build_phase_envelope();
    /// assert!(res.is_ok());
    /// # Ok::<(), rfluids::Error>(())
    /// ```
    ///
    /// # See Also
    ///
    /// - [`Fluid::phase_envelope_data`]
    pub fn build_phase_envelope(&mut self) -> StateResult<&mut Self> {
        self.backend.build_phase_envelope()?;
        Ok(self)
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
    ///   component count if known (e.g. a [`CustomMix`]'s)
    ///
    /// # Errors
    ///
    /// Returns a [`FluidOutputError::PhaseEnvelopeUnavailable`] if `CoolProp` cannot report the
    /// trace, if the trace filled the entire `max_points` capacity, or if it reports more
    /// components than `max_components` can hold.
    ///
    /// # Examples
    ///
    /// ```
    /// use rfluids::prelude::*;
    ///
    /// let mut mixture: Fluid<Undefined> =
    ///     CustomMix::mole_based([(Pure::Nitrogen, 0.79), (Pure::Oxygen, 0.21)])?.try_into()?;
    /// mixture.build_phase_envelope()?;
    /// let trace = mixture.phase_envelope_data(2000, 2)?;
    /// assert!(!trace.temperature.is_empty());
    /// # Ok::<(), rfluids::Error>(())
    /// ```
    ///
    /// # See Also
    ///
    /// - [`Fluid::build_phase_envelope`]
    pub fn phase_envelope_data(
        &mut self,
        max_points: usize,
        max_components: usize,
    ) -> OutputResult<PhaseEnvelopeData> {
        self.backend
            .phase_envelope_data(max_points, max_components)
            .map_err(FluidOutputError::PhaseEnvelopeUnavailable)
    }

    /// Acentric factor **\[dimensionless\]**.
    ///
    /// # Errors
    ///
    /// Returns a [`FluidOutputError`] if the property is not available for the specified substance.
    pub fn acentric_factor(&mut self) -> OutputResult<f64> {
        self.trivial_output(FluidTrivialParam::AcentricFactor)
    }

    /// Critical point mass density **\[kg/m³\]**.
    ///
    /// # Errors
    ///
    /// Returns a [`FluidOutputError`] if the property is not available for the specified substance.
    pub fn critical_density(&mut self) -> OutputResult<f64> {
        let key = FluidTrivialParam::DMassCritical;
        // Due to CoolProp freeze
        if let Substance::PredefinedMix(_) = self.substance {
            return Err(FluidOutputError::UnavailableTrivialOutput(key));
        }
        self.positive_trivial_output(key)
    }

    /// Critical point molar density **\[mol/m³\]**.
    ///
    /// # Errors
    ///
    /// Returns a [`FluidOutputError`] if the property is not available for the specified substance.
    pub fn critical_molar_density(&mut self) -> OutputResult<f64> {
        let key = FluidTrivialParam::DMolarCritical;
        // Due to CoolProp freeze
        if let Substance::PredefinedMix(_) = self.substance {
            return Err(FluidOutputError::UnavailableTrivialOutput(key));
        }
        self.positive_trivial_output(key)
    }

    /// Critical point pressure **\[Pa\]**.
    ///
    /// # Errors
    ///
    /// Returns a [`FluidOutputError`] if the property is not available for the specified substance.
    pub fn critical_pressure(&mut self) -> OutputResult<f64> {
        let key = FluidTrivialParam::PCritical;
        // Due to CoolProp freeze
        if let Substance::PredefinedMix(_) = self.substance {
            return Err(FluidOutputError::UnavailableTrivialOutput(key));
        }
        self.positive_trivial_output(key)
    }

    /// Critical point temperature **\[K\]**.
    ///
    /// # Errors
    ///
    /// Returns a [`FluidOutputError`] if the property is not available for the specified substance.
    pub fn critical_temperature(&mut self) -> OutputResult<f64> {
        let key = FluidTrivialParam::TCritical;
        // Due to CoolProp freeze
        if let Substance::PredefinedMix(_) = self.substance {
            return Err(FluidOutputError::UnavailableTrivialOutput(key));
        }
        self.positive_trivial_output(key)
    }

    /// Flammability hazard index **\[dimensionless\]**.
    ///
    /// # Errors
    ///
    /// Returns a [`FluidOutputError`] if the property is not available for the specified substance.
    pub fn flammability_hazard(&mut self) -> OutputResult<f64> {
        self.non_negative_trivial_output(FluidTrivialParam::FH)
    }

    /// Freezing temperature for incompressible mixtures **\[K\]**.
    ///
    /// # Errors
    ///
    /// Returns a [`FluidOutputError`] if the property is not available for the specified substance.
    pub fn freezing_temperature(&mut self) -> OutputResult<f64> {
        self.positive_trivial_output(FluidTrivialParam::TFreeze)
    }

    /// 20-year global warming potential **\[dimensionless\]**.
    ///
    /// # Errors
    ///
    /// Returns a [`FluidOutputError`] if the property is not available for the specified substance.
    pub fn gwp20(&mut self) -> OutputResult<f64> {
        self.non_negative_trivial_output(FluidTrivialParam::GWP20)
    }

    /// 100-year global warming potential **\[dimensionless\]**.
    ///
    /// # Errors
    ///
    /// Returns a [`FluidOutputError`] if the property is not available for the specified substance.
    pub fn gwp100(&mut self) -> OutputResult<f64> {
        self.non_negative_trivial_output(FluidTrivialParam::GWP100)
    }

    /// 500-year global warming potential **\[dimensionless\]**.
    ///
    /// # Errors
    ///
    /// Returns a [`FluidOutputError`] if the property is not available for the specified substance.
    pub fn gwp500(&mut self) -> OutputResult<f64> {
        self.non_negative_trivial_output(FluidTrivialParam::GWP500)
    }

    /// Health hazard index **\[dimensionless\]**.
    ///
    /// # Errors
    ///
    /// Returns a [`FluidOutputError`] if the property is not available for the specified substance.
    pub fn health_hazard(&mut self) -> OutputResult<f64> {
        self.non_negative_trivial_output(FluidTrivialParam::HH)
    }

    /// Maximum pressure **\[Pa\]**.
    ///
    /// # Errors
    ///
    /// Returns a [`FluidOutputError`] if the property is not available for the specified substance.
    pub fn max_pressure(&mut self) -> OutputResult<f64> {
        self.positive_trivial_output(FluidTrivialParam::PMax)
    }

    /// Maximum temperature **\[K\]**.
    pub fn max_temperature(&mut self) -> f64 {
        self.positive_trivial_output(FluidTrivialParam::TMax).unwrap()
    }

    /// Minimum pressure **\[Pa\]**.
    ///
    /// # Errors
    ///
    /// Returns a [`FluidOutputError`] if the property is not available for the specified substance.
    pub fn min_pressure(&mut self) -> OutputResult<f64> {
        self.positive_trivial_output(FluidTrivialParam::PMin)
    }

    /// Minimum temperature **\[K\]**.
    pub fn min_temperature(&mut self) -> f64 {
        self.positive_trivial_output(FluidTrivialParam::TMin).unwrap()
    }

    /// Molar mass **\[kg/mol\]**.
    ///
    /// # Errors
    ///
    /// Returns a [`FluidOutputError`] if the property is not available for the specified substance.
    pub fn molar_mass(&mut self) -> OutputResult<f64> {
        self.positive_trivial_output(FluidTrivialParam::MolarMass)
    }

    /// Current mole fractions **\[dimensionless, from 0 to 1 each\]**, keyed by component.
    ///
    /// Only available for [`CustomMix`](crate::substance::CustomMix)-backed fluids -- for any
    /// other substance, the composition is already fully described by
    /// [`Fluid::substance`](crate::fluid::Fluid::substance) (e.g. a single
    /// [`Pure`](crate::substance::Pure), or a [`BinaryMix`](crate::substance::BinaryMix)'s
    /// already-labeled `fraction`), so there is no "which value is which" ambiguity to resolve.
    ///
    /// # Errors
    ///
    /// Returns a [`FluidOutputError::NotACustomMix`] if the current substance isn't a
    /// [`CustomMix`](crate::substance::CustomMix), or [`FluidOutputError::CompositionUnavailable`]
    /// if `CoolProp` cannot otherwise report the composition.
    ///
    /// # Examples
    ///
    /// ```
    /// use rfluids::prelude::*;
    ///
    /// let mut mixture: Fluid<Undefined> =
    ///     CustomMix::mole_based([(Pure::Water, 0.8), (Pure::Ethanol, 0.2)])?.try_into()?;
    /// let res = mixture.mole_fractions()?;
    /// assert_eq!(res[&Pure::Water], 0.8);
    /// assert_eq!(res[&Pure::Ethanol], 0.2);
    /// # Ok::<(), rfluids::Error>(())
    /// ```
    ///
    /// # See Also
    ///
    /// - [`Fluid::mole_fractions_sat_state`](crate::fluid::Fluid::mole_fractions_sat_state)
    pub fn mole_fractions(&mut self) -> OutputResult<HashMap<Pure, f64>> {
        let order =
            custom_mix_component_order(&self.substance).ok_or(FluidOutputError::NotACustomMix)?;
        let raw = self
            .backend
            .mole_fractions(order.len())
            .map_err(FluidOutputError::CompositionUnavailable)?;
        Ok(zip_component_order(order, raw))
    }

    /// Ozone depletion potential **\[dimensionless\]**.
    ///
    /// # Errors
    ///
    /// Returns a [`FluidOutputError`] if the property is not available for the specified substance.
    pub fn odp(&mut self) -> OutputResult<f64> {
        self.non_negative_trivial_output(FluidTrivialParam::ODP)
    }

    /// Physical hazard index **\[dimensionless\]**.
    ///
    /// # Errors
    ///
    /// Returns a [`FluidOutputError`] if the property is not available for the specified substance.
    pub fn physical_hazard(&mut self) -> OutputResult<f64> {
        self.non_negative_trivial_output(FluidTrivialParam::PH)
    }

    /// Reducing point mass density **\[kg/m³\]**.
    ///
    /// # Errors
    ///
    /// Returns a [`FluidOutputError`] if the property is not available for the specified substance.
    pub fn reducing_density(&mut self) -> OutputResult<f64> {
        mul(self.reducing_molar_density(), self.molar_mass())
    }

    /// Reducing point molar density **\[mol/m³\]**.
    ///
    /// # Errors
    ///
    /// Returns a [`FluidOutputError`] if the property is not available for the specified substance.
    pub fn reducing_molar_density(&mut self) -> OutputResult<f64> {
        self.positive_trivial_output(FluidTrivialParam::DMolarReducing)
    }

    /// Reducing point pressure **\[Pa\]**.
    ///
    /// # Errors
    ///
    /// Returns a [`FluidOutputError`] if the property is not available for the specified substance.
    pub fn reducing_pressure(&mut self) -> OutputResult<f64> {
        self.positive_trivial_output(FluidTrivialParam::PReducing)
    }

    /// Reducing point temperature **\[K\]**.
    ///
    /// # Errors
    ///
    /// Returns a [`FluidOutputError`] if the property is not available for the specified substance.
    pub fn reducing_temperature(&mut self) -> OutputResult<f64> {
        self.positive_trivial_output(FluidTrivialParam::TReducing)
    }

    /// Triple point pressure **\[Pa\]**.
    ///
    /// # Errors
    ///
    /// Returns a [`FluidOutputError`] if the property is not available for the specified substance.
    pub fn triple_pressure(&mut self) -> OutputResult<f64> {
        self.positive_trivial_output(FluidTrivialParam::PTriple)
    }

    /// Triple point temperature **\[K\]**.
    ///
    /// # Errors
    ///
    /// Returns a [`FluidOutputError`] if the property is not available for the specified substance.
    pub fn triple_temperature(&mut self) -> OutputResult<f64> {
        self.positive_trivial_output(FluidTrivialParam::TTriple)
    }

    pub(crate) fn inner_specify_phase(&mut self, phase: Phase) -> Result<(), FluidPhaseError> {
        if phase == Phase::NotImposed {
            self.inner_unspecify_phase();
            return Ok(());
        }
        self.backend.specify_phase(phase)?;
        self.specified_phase = phase;
        Ok(())
    }

    pub(crate) fn inner_unspecify_phase(&mut self) {
        self.backend.unspecify_phase();
        self.specified_phase = Phase::NotImposed;
    }

    pub(crate) fn inner_update(
        &mut self,
        input1: FluidInput,
        input2: FluidInput,
    ) -> StateResult<()> {
        let request: FluidUpdateRequest = (input1, input2).try_into()?;
        self.backend.update(request.input_pair, request.value1, request.value2)?;
        self.outputs.clear();
        self.outputs.insert(input1.key, Ok(input1.value));
        self.outputs.insert(input2.key, Ok(input2.value));
        self.update_request = Some(request);
        Ok(())
    }

    /// Updates a [`CustomMix`](crate::substance::CustomMix)-backed fluid's mole fractions in
    /// place, keeping its components (and the underlying `CoolProp` backend) otherwise
    /// unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`FluidStateError::IncompatibleComponents`] if the current substance isn't a
    /// [`CustomMix`](crate::substance::CustomMix) with the same components as `components`.
    pub(crate) fn inner_update_mole_fractions(&mut self, components: CustomMix) -> StateResult<()> {
        let current_order = custom_mix_component_order(&self.substance)
            .ok_or(FluidStateError::IncompatibleComponents)?;
        let new = components.into_mole_based();
        let new_components = new.components();
        if new_components.len() != current_order.len() {
            return Err(FluidStateError::IncompatibleComponents);
        }
        let fractions = current_order
            .iter()
            .map(|pure| {
                new_components.get(pure).copied().ok_or(FluidStateError::IncompatibleComponents)
            })
            .collect::<Result<Vec<f64>, FluidStateError>>()?;
        self.backend.set_fractions(&fractions)?;
        self.substance = Substance::CustomMix(new);
        self.outputs.clear();
        self.trivial_outputs.clear();
        Ok(())
    }

    /// Updates a [`BinaryMix`](crate::substance::BinaryMix)-backed fluid's fraction in place,
    /// keeping its kind (and the underlying `CoolProp` backend) otherwise unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`FluidStateError::IncompatibleComponents`] if the current substance isn't a
    /// [`BinaryMix`](crate::substance::BinaryMix) of the same
    /// [`kind`](crate::substance::BinaryMix::kind) as `mix`.
    pub(crate) fn inner_update_fraction(&mut self, mix: BinaryMix) -> StateResult<()> {
        let Substance::BinaryMix(current) = &self.substance else {
            return Err(FluidStateError::IncompatibleComponents);
        };
        if current.kind != mix.kind {
            return Err(FluidStateError::IncompatibleComponents);
        }
        self.backend.set_fractions(&[mix.fraction])?;
        self.substance = Substance::BinaryMix(mix);
        self.outputs.clear();
        self.trivial_outputs.clear();
        Ok(())
    }

    fn positive_trivial_output(&mut self, key: FluidTrivialParam) -> OutputResult<f64> {
        self.trivial_output(key).and_then(|value| guard(key.into(), value, |x| x > 0.0))
    }

    fn non_negative_trivial_output(&mut self, key: FluidTrivialParam) -> OutputResult<f64> {
        self.trivial_output(key).and_then(|value| guard(key.into(), value, |x| x >= 0.0))
    }

    fn trivial_output(&mut self, key: FluidTrivialParam) -> OutputResult<f64> {
        cached_output(&mut self.trivial_outputs, &mut self.backend, key, |_| {
            FluidOutputError::UnavailableTrivialOutput(key)
        })
        .and_then(|value| guard(key.into(), value, f64::is_finite))
    }
}

/// Ordered [`Pure`] components of `substance`, in the fixed order a `CoolProp` backend built
/// from it would use (see [`CustomMix::sorted_by_name`]) -- `None` if `substance` isn't a
/// [`CustomMix`].
///
/// Shared by [`Fluid::inner_update_mole_fractions`](Fluid::inner_update_mole_fractions),
/// [`Fluid::mole_fractions`](crate::fluid::Fluid::mole_fractions), and
/// [`Fluid::mole_fractions_sat_state`](crate::fluid::Fluid::mole_fractions_sat_state) so all
/// three agree on the same order derived the same way.
pub(crate) fn custom_mix_component_order(substance: &Substance) -> Option<Vec<Pure>> {
    let Substance::CustomMix(mix) = substance else {
        return None;
    };
    Some(mix.clone().into_mole_based().sorted_by_name().into_iter().map(|(pure, _)| pure).collect())
}

/// Pairs `order` (from [`custom_mix_component_order`]) positionally with a fraction vector
/// `CoolProp` reported for that same order (e.g. from
/// [`AbstractState::mole_fractions`](crate::native::AbstractState::mole_fractions)).
///
/// # Panics
///
/// Panics if the lengths don't match -- an invariant violation (a bug in `rfluids`, not
/// something a caller's input can trigger), since both are derived from the same substance.
pub(crate) fn zip_component_order(order: Vec<Pure>, fractions: Vec<f64>) -> HashMap<Pure, f64> {
    assert_eq!(
        order.len(),
        fractions.len(),
        "CoolProp reported {} mole fraction(s) for a {}-component CustomMix",
        fractions.len(),
        order.len()
    );
    order.into_iter().zip(fractions).collect()
}
