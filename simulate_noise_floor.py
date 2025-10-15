#!/usr/bin/env python3
"""
Simulate a noise floor measurement based on stanza.routines.builtins.health_check
"""

import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Simulation parameters
num_points = 100
true_mean = 1e-12  # 1 pA baseline current
true_std = 5e-13   # 0.5 pA noise level

# Generate simulated noise floor measurements
# Model as Gaussian noise around a baseline current
currents = np.random.normal(true_mean, true_std, num_points)

# Calculate statistics (as done in noise_floor_measurement routine)
current_mean = np.mean(currents)
current_std = np.std(currents)

print("Noise Floor Measurement Results:")
print(f"  Number of points: {num_points}")
print(f"  Mean current: {current_mean:.4e} A ({current_mean * 1e12:.4f} pA)")
print(f"  Std deviation: {current_std:.4e} A ({current_std * 1e12:.4f} pA)")
print(f"  Mean + 1σ: {(current_mean + current_std) * 1e12:.4f} pA")
print(f"  Mean - 1σ: {(current_mean - current_std) * 1e12:.4f} pA")

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Create time axis from 0 to 5 seconds
time_points = np.linspace(0, 5, num_points)

# Plot measurement points
ax.plot(time_points, currents * 1e12, 'o', color='gray',
        alpha=1.0, markersize=5, label='Noise floor measurements')

# Plot mean as solid horizontal line
ax.axhline(current_mean * 1e12, color='blue', linestyle='-',
           linewidth=2, label=f'Mean ({current_mean * 1e12:.4f} pA)')

# Plot ±1 standard deviation bounds as dotted lines
ax.axhline((current_mean + current_std) * 1e12, color='red',
           linestyle=':', linewidth=2, label=f'Mean + 1σ ({(current_mean + current_std) * 1e12:.4f} pA)')
ax.axhline((current_mean - current_std) * 1e12, color='red',
           linestyle=':', linewidth=2, label=f'Mean - 1σ ({(current_mean - current_std) * 1e12:.4f} pA)')

# Labels and formatting
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Current (pA)', fontsize=12)
ax.set_title('Noise Floor Measurement', fontsize=14, pad=20)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left', fontsize=10)

# Set x-axis limits
ax.set_xlim(0, 5)

plt.tight_layout(pad=2.0)

# Save the figure
output_file = '/Users/rjow/cq/stanza/noise_floor_measurement.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=1.0)
print(f"\nPlot saved to: {output_file}")

# Show the plot
plt.show()
