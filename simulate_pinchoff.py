#!/usr/bin/env python3
"""
Simulate a pinchoff curve with noise and fit it using stanza.analysis.fitting
"""

import numpy as np
import matplotlib.pyplot as plt
from stanza.analysis.fitting import fit_pinchoff_parameters, pinchoff_curve

# Set random seed for reproducibility
np.random.seed(42)

# Generate simulated pinchoff curve data
voltages = np.linspace(-0.5, 1.5, 200)

# True parameters for the pinchoff curve (in normalized space approximately)
# We'll generate data that looks like a typical pinchoff curve
true_a = 5e-9  # Amplitude (nA scale)
true_b = 8.0   # Slope
true_c = -3.6  # Offset (transition at ~0.45V)

# Generate ideal curve
ideal_currents = true_a * (1 + np.tanh(true_b * voltages + true_c))

# Add realistic noise (combination of absolute and relative noise)
noise_level = 0.05 * true_a  # 5% of amplitude
noise = np.random.normal(0, noise_level, size=voltages.shape)
currents = ideal_currents + noise

# Ensure currents are non-negative (physical constraint)
currents = np.maximum(currents, 1e-12)

# Fit the pinchoff curve
print("Fitting pinchoff curve...")
fit_result = fit_pinchoff_parameters(voltages, currents, sigma=2.0)

print(f"\nFit Results:")
print(f"  v_cut_off (red):     {fit_result.v_cut_off:.4f} V")
print(f"  v_transition (green): {fit_result.v_transition:.4f} V")
print(f"  v_saturation (blue):  {fit_result.v_saturation:.4f} V")
print(f"  Fitted parameters: a={fit_result.popt[0]:.4e}, b={fit_result.popt[1]:.4f}, c={fit_result.popt[2]:.4f}")

# Generate fitted curve for plotting
fitted_currents = fit_result.fit_curve(voltages)

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot simulated data with noise (grey points)
ax.plot(voltages, currents * 1e9, 'o', color='gray', alpha=1.0,
        markersize=4, label='Pinch off sweep data')

# Plot fitted curve (black line)
ax.plot(voltages, fitted_currents * 1e9, 'k-', linewidth=2, label='Fitted curve')

# Add vertical lines for characteristic voltages
if fit_result.v_cut_off is not None:
    ax.axvline(fit_result.v_cut_off, color='blue', linestyle='--',
               linewidth=2, label=f'Cut-off voltage ({fit_result.v_cut_off:.3f} V)')

if fit_result.v_transition is not None:
    ax.axvline(fit_result.v_transition, color='green', linestyle='--',
               linewidth=2, label=f'Transition voltage ({fit_result.v_transition:.3f} V)')

if fit_result.v_saturation is not None:
    ax.axvline(fit_result.v_saturation, color='red', linestyle='--',
               linewidth=2, label=f'Saturation voltage ({fit_result.v_saturation:.3f} V)')

# Labels and formatting
ax.set_xlabel('Gate Voltage (V)', fontsize=12)
ax.set_ylabel('Current (nA)', fontsize=12)
ax.set_title('Global Accumulation', fontsize=14, pad=20)
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=10)

plt.tight_layout(pad=2.0)

# Save the figure
output_file = '/Users/rjow/cq/stanza/global_accumulation.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=1.0)
print(f"\nPlot saved to: {output_file}")

# Show the plot
plt.show()
