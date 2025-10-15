#!/usr/bin/env python3
"""
Simulate finger gate characterization with multiple pinchoff curves
Based on stanza.routines.builtins.health_check finger_gate_characterization
"""

import numpy as np
import matplotlib.pyplot as plt
from stanza.analysis.fitting import fit_pinchoff_parameters

# Set random seed for reproducibility
np.random.seed(42)

# Define multiple finger gates with different characteristics
# Some gates have similar transition voltages (horizontally aligned)
# Cut-off voltages clustered between 0.2 and 0.4 V
finger_gates = [
    {'name': 'P1', 'color': 'blue', 'light_color': 'lightblue',
     'v_transition': 0.28, 'amplitude': 6e-9, 'noise': 0.04},
    {'name': 'B1', 'color': 'green', 'light_color': 'lightgreen',
     'v_transition': 0.30, 'amplitude': 5e-9, 'noise': 0.05},
    {'name': 'P2', 'color': 'purple', 'light_color': 'plum',
     'v_transition': 0.35, 'amplitude': 7e-9, 'noise': 0.045},
    {'name': 'B2', 'color': 'orange', 'light_color': 'lightsalmon',
     'v_transition': 0.38, 'amplitude': 4.5e-9, 'noise': 0.06},
]

# Create the plot
fig, ax = plt.subplots(figsize=(12, 7))

# Store fit results
fit_results = []

print("Fitting finger gate characterization curves...")

for gate in finger_gates:
    # Generate voltage range (all start at 0)
    voltages = np.linspace(0.0, 1.0, 200)

    # Calculate parameters for desired transition voltage
    true_a = gate['amplitude']
    true_b = 8.0
    true_c = -true_b * gate['v_transition']  # Center transition at v_transition

    # Generate ideal curve
    ideal_currents = true_a * (1 + np.tanh(true_b * voltages + true_c))

    # Add noise
    noise_level = gate['noise'] * true_a
    noise = np.random.normal(0, noise_level, size=voltages.shape)
    currents = ideal_currents + noise
    currents = np.maximum(currents, 1e-12)

    # Fit the curve
    fit_result = fit_pinchoff_parameters(voltages, currents, sigma=2.0)
    fit_results.append({'name': gate['name'], 'fit': fit_result})

    print(f"\n{gate['name']} Results:")
    print(f"  v_cut_off: {fit_result.v_cut_off:.4f} V")
    print(f"  v_transition: {fit_result.v_transition:.4f} V")
    print(f"  v_saturation: {fit_result.v_saturation:.4f} V")

    # Generate fitted curve
    fitted_currents = fit_result.fit_curve(voltages)

    # Plot data and fit
    ax.plot(voltages, currents * 1e9, 'o', color=gate['light_color'], alpha=1.0,
            markersize=3, label=f'{gate["name"]} sweep data')
    ax.plot(voltages, fitted_currents * 1e9, '-', color=gate['color'],
            linewidth=2, label=f'{gate["name"]} fitted curve')

    # Add vertical line for cut-off voltage
    if fit_result.v_cut_off is not None:
        ax.axvline(fit_result.v_cut_off, color=gate['color'], linestyle='--',
                   linewidth=2, alpha=0.7,
                   label=f'{gate["name"]} cut-off ({fit_result.v_cut_off:.3f} V)')

# Labels and formatting
ax.set_xlabel('Gate Voltage (V)', fontsize=12)
ax.set_ylabel('Current (nA)', fontsize=12)
ax.set_title('Finger Gate Characterization', fontsize=14, pad=20)
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=9, ncol=1)

# Set axis limits to start at 0
ax.set_xlim(0, None)
ax.set_ylim(0, None)

plt.tight_layout(pad=2.0)

# Save the figure
output_file = '/Users/rjow/cq/stanza/finger_gate_characterization.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=1.0)
print(f"\nPlot saved to: {output_file}")

# Show the plot
plt.show()
