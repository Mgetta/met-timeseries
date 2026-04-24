"""Scientific and thermodynamic constants for meteorological derivations."""

# Fundamental Physics
STEFAN_BOLTZMANN = 5.67e-8  # Stefan-Boltzmann constant (W/m²/K⁴)
EPSILON = 0.622             # Ratio of molecular weight of water vapor to dry air

# Shared Thermodynamic Approximations
SAT_VP_0C_HPA = 6.112       # Saturation vapor pressure of water at 0°C in hPa (or mb)
VAPOR_A_MAGNUS = 17.27      # Constant 'A' for Magnus-Tetens vapor pressure formulas
VAPOR_B_MAGNUS = 237.3      # Constant 'B' (°C) for standard Magnus formula (FAO-56)
VAPOR_B_TETENS = 237.7      # Constant 'B' (°C) for August-Roche-Magnus approximation