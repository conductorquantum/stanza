#!/usr/bin/env python3
"""
Simulate reservoir characterization with multiple pinchoff curves
Based on stanza.routines.builtins.health_check reservoir_characterization
"""

import numpy as np
import matplotlib.pyplot as plt
from stanza.analysis.fitting import fit_pinchoff_parameters

# Set random seed for reproducibility
np.random.seed(42)

# Reservoir names
reservoir_names = ['R1', 'R2']

# Generate simulated pinchoff curve data for two reservoirs
# Reservoir 1: centered around 0.3V
voltages_r1 = np.linspace(0.0, 1.0, 200)
true_a_r1 = 5e-9
true_b_r1 = 8.0
true_c_r1 = -2.4  # Offset for transition at ~0.3V

ideal_currents_r1 = true_a_r1 * (1 + np.tanh(true_b_r1 * voltages_r1 + true_c_r1))
noise_level_r1 = 0.05 * true_a_r1
noise_r1 = np.random.normal(0, noise_level_r1, size=voltages_r1.shape)
currents_r1 = ideal_currents_r1 + noise_r1
currents_r1 = np.maximum(currents_r1, 1e-12)

# Reservoir 2: centered around 0.55V (horizontal offset)
voltages_r2 = np.linspace(0.0, 1.0, 200)
true_a_r2 = 5e-9
true_b_r2 = 8.0
true_c_r2 = -4.4  # Offset for transition at ~0.55V

ideal_currents_r2 = true_a_r2 * (1 + np.tanh(true_b_r2 * voltages_r2 + true_c_r2))
noise_level_r2 = 0.05 * true_a_r2
noise_r2 = np.random.normal(0, noise_level_r2, size=voltages_r2.shape)
currents_r2 = ideal_currents_r2 + noise_r2
currents_r2 = np.maximum(currents_r2, 1e-12)

# Fit both reservoirs
print("Fitting reservoir characterization curves...")
fit_result_r1 = fit_pinchoff_parameters(voltages_r1, currents_r1, sigma=2.0)
fit_result_r2 = fit_pinchoff_parameters(voltages_r2, currents_r2, sigma=2.0)

print(f"\nReservoir 1 (R1) Results:")
print(f"  v_cut_off: {fit_result_r1.v_cut_off:.4f} V")
print(f"  v_transition: {fit_result_r1.v_transition:.4f} V")
print(f"  v_saturation: {fit_result_r1.v_saturation:.4f} V")

print(f"\nReservoir 2 (R2) Results:")
print(f"  v_cut_off: {fit_result_r2.v_cut_off:.4f} V")
print(f"  v_transition: {fit_result_r2.v_transition:.4f} V")
print(f"  v_saturation: {fit_result_r2.v_saturation:.4f} V")

# Generate fitted curves
fitted_currents_r1 = fit_result_r1.fit_curve(voltages_r1)
fitted_currents_r2 = fit_result_r2.fit_curve(voltages_r2)

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot R1 data and fit
ax.plot(voltages_r1, currents_r1 * 1e9, 'o', color='lightblue', alpha=1.0,
        markersize=4, label='Reservoir Gate 1 sweep data')
ax.plot(voltages_r1, fitted_currents_r1 * 1e9, '-', color='blue',
        linewidth=2, label='Reservoir Gate 1 fitted curve')

# Plot R2 data and fit
ax.plot(voltages_r2, currents_r2 * 1e9, 'o', color='lightgreen', alpha=1.0,
        markersize=4, label='Reservoir Gate 2 sweep data')
ax.plot(voltages_r2, fitted_currents_r2 * 1e9, '-', color='green',
        linewidth=2, label='Reservoir Gate 2 fitted curve')

# Add vertical lines for cut-off voltages with different colors
if fit_result_r1.v_cut_off is not None:
    ax.axvline(fit_result_r1.v_cut_off, color='darkblue', linestyle='--',
               linewidth=2, label=f'Reservoir Gate 1 cut-off ({fit_result_r1.v_cut_off:.3f} V)')

if fit_result_r2.v_cut_off is not None:
    ax.axvline(fit_result_r2.v_cut_off, color='darkgreen', linestyle='--',
               linewidth=2, label=f'Reservoir Gate 2 cut-off ({fit_result_r2.v_cut_off:.3f} V)')

# Labels and formatting
ax.set_xlabel('Gate Voltage (V)', fontsize=12)
ax.set_ylabel('Current (nA)', fontsize=12)
ax.set_title('Reservoir Characterization', fontsize=14, pad=20)
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=10)

# Set axis limits to start at 0
ax.set_xlim(0, None)
ax.set_ylim(0, None)

plt.tight_layout(pad=2.0)

# Save the figure
output_file = '/Users/rjow/cq/stanza/reservoir_characterization.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=1.0)
print(f"\nPlot saved to: {output_file}")

# Show the plot
plt.show()
