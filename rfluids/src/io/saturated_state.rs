/// Which side of a two-phase equilibrium state's saturation dome to read.
///
/// Used by
/// [`AbstractState::mole_fractions_sat_state`](crate::native::AbstractState::mole_fractions_sat_state)
/// and [`Fluid::mole_fractions_sat_state`](crate::fluid::Fluid::mole_fractions_sat_state) to
/// pick between the liquid and gas sides of the current (already two-phase) equilibrium state --
/// e.g. the `x`/`y` compositions of a flash result.
///
/// # Note
///
/// `CoolPropLib.h` documents `AbstractState_get_mole_fractions_satState`'s `saturated_state`
/// argument only by its type (`const char *`). The wire values below (`"liquid"`/`"gas"`) were
/// confirmed empirically against a known two-phase point: both are accepted and return the
/// physically-expected split, while `"vapor"` -- a plausible-looking spelling for the gas side
/// -- is rejected.
///
/// # Examples
///
/// Conversion between [`&str`](str):
///
/// ```
/// use std::str::FromStr;
///
/// use rfluids::prelude::*;
///
/// assert_eq!(SaturatedState::Liquid.as_ref(), "liquid");
/// assert_eq!(SaturatedState::from_str("liquid"), Ok(SaturatedState::Liquid));
/// ```
#[derive(
    Copy,
    Clone,
    Debug,
    Eq,
    PartialEq,
    strum_macros::AsRefStr,
    strum_macros::EnumString,
    strum_macros::IntoStaticStr,
)]
#[strum(ascii_case_insensitive)]
pub enum SaturatedState {
    /// Saturated-liquid side (the `x` composition of a two-phase result).
    #[strum(to_string = "liquid")]
    Liquid,

    /// Saturated-vapor side (the `y` composition of a two-phase result).
    ///
    /// `CoolProp` spells this `"gas"`, not `"vapor"`.
    #[strum(to_string = "gas")]
    Gas,
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use rstest::*;

    use super::{SaturatedState::*, *};

    #[rstest]
    #[case(Liquid, "liquid")]
    #[case(Gas, "gas")]
    fn as_str(#[case] sut: SaturatedState, #[case] expected: &str) {
        // When
        let str = sut.as_ref();
        let static_str: &'static str = sut.into();

        // Then
        assert_eq!(str, expected);
        assert_eq!(static_str, expected);
    }

    #[rstest]
    #[case(vec!["liquid", "Liquid", "LIQUID"], Liquid)]
    #[case(vec!["gas", "Gas", "GAS"], Gas)]
    fn from_valid_str(#[case] valid: Vec<&str>, #[case] expected: SaturatedState) {
        for s in valid {
            // When
            let res1 = SaturatedState::from_str(s);
            let res2 = SaturatedState::try_from(s);

            // Then
            assert_eq!(res1, Ok(expected));
            assert_eq!(res2, Ok(expected));
        }
    }

    #[rstest]
    #[case("")]
    #[case("vapor")]
    #[case("phase_liquid")]
    fn from_invalid_str(#[case] invalid: &str) {
        // When
        let res = SaturatedState::from_str(invalid);

        // Then
        assert!(res.is_err());
    }
}
