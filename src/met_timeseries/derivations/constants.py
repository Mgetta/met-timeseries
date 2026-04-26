"""Scientific and thermodynamic constants for meteorological derivations."""

# Fundamental Physics
STEFAN_BOLTZMANN = 5.67e-8  # Stefan-Boltzmann constant (W/m²/K⁴)
EPSILON = 0.622             # Ratio of molecular weight of water vapor to dry air
SOLAR_CONSTANT_W_M2 = 1361.0  # Solar constant (W/m²) at Earth's distance

# --- Solar ---
SOLAR_CONSTANT_W_M2 = 1361.0    # Solar constant (W/m²)
SOLAR_CONSTANT_MJ_DAY = 117.5   # Solar constant (MJ/m²/day)


# --- Thermodynamic ---
PSYCHROMETRIC_COEFFICIENT = 0.000665  # γ = cp / (λ·ε) (kPa/°C per kPa pressure)
LAMBDA_0 = 2.501                      # Latent heat of vaporisation at 0 °C (MJ/kg)
LAMBDA_T = 0.002361                   # Temperature correction for λ (MJ/kg/°C)
RECIPROCAL_LAMBDA_20C = 0.408         # 1/λ at ~20 °C, FAO-56 shorthand (kg/MJ)