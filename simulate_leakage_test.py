#!/usr/bin/env python3
"""
Simulate a leakage test matrix based on stanza.routines.builtins.health_check
"""

import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Simulation parameters
num_gates = 7
gate_names = [f'G{i+1}' for i in range(num_gates)]

# Generate symmetric leakage resistance matrix (in Ohms)
# Most resistances should be on the order of 10^13 Ohms (good isolation)
# Scale: 1e6 to 1e13 Ohms

# Initialize with random resistances across different orders of magnitude
leakage_matrix = np.zeros((num_gates, num_gates))

# Define different resistance ranges (orders of magnitude)
resistance_ranges = [
    (8e12, 1e13),  # Good isolation
    (5e11, 1e12),  # Moderate
    (1e10, 5e10),  # Some leakage
    (5e9, 1e10),   # More leakage
    (1e9, 5e9),    # Significant leakage
    (1e8, 5e8),    # High leakage
    (1e6, 50e6),   # Failed connection (<50 MOhm)
]

for i in range(num_gates):
    for j in range(i, num_gates):
        if i == j:
            # Diagonal: self-measurement resistance (high but finite)
            leakage_matrix[i, j] = np.random.uniform(8e12, 1e13)
        else:
            # Off-diagonal: gate-to-gate resistance - mix of different orders
            # Most connections are good (10^12-10^13), some have varying degrees of leakage
            rand_val = np.random.random()
            if rand_val < 0.55:  # 55% good isolation
                leakage_matrix[i, j] = np.random.uniform(*resistance_ranges[0])
            elif rand_val < 0.70:  # 15% moderate
                leakage_matrix[i, j] = np.random.uniform(*resistance_ranges[1])
            elif rand_val < 0.82:  # 12% some leakage
                leakage_matrix[i, j] = np.random.uniform(*resistance_ranges[2])
            elif rand_val < 0.89:  # 7% more leakage
                leakage_matrix[i, j] = np.random.uniform(*resistance_ranges[3])
            elif rand_val < 0.93:  # 4% significant leakage
                leakage_matrix[i, j] = np.random.uniform(*resistance_ranges[4])
            elif rand_val < 0.96:  # 3% high leakage
                leakage_matrix[i, j] = np.random.uniform(*resistance_ranges[5])
            else:  # 4% failed (<50 MOhm)
                leakage_matrix[i, j] = np.random.uniform(*resistance_ranges[6])

# Make the matrix symmetric by copying upper triangle to lower triangle
for i in range(num_gates):
    for j in range(i+1, num_gates):
        leakage_matrix[j, i] = leakage_matrix[i, j]

print("Leakage Test Results:")
print(f"  Number of gates: {num_gates}")
print(f"  Gate names: {', '.join(gate_names)}")
print(f"\nLeakage Resistance Matrix (Ohms):")
print("=" * 80)

# Print the matrix in a readable format
header = "      " + "".join([f"{name:>12}" for name in gate_names])
print(header)
print("-" * 80)
for i, gate_i in enumerate(gate_names):
    row_str = f"{gate_i:>4}  "
    for j in range(num_gates):
        row_str += f"{leakage_matrix[i, j]:.2e} "
    print(row_str)

# Create the heatmap plot
fig, ax = plt.subplots(figsize=(10, 8))

# Create heatmap with log scale (vmin=1e6, vmax=1e13)
im = ax.imshow(leakage_matrix, cmap='Blues', aspect='auto',
               norm=plt.matplotlib.colors.LogNorm(vmin=1e6, vmax=1e13))

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Resistance (Ω)', fontsize=12, rotation=270, labelpad=20)

# Set ticks and labels
ax.set_xticks(np.arange(num_gates))
ax.set_yticks(np.arange(num_gates))
ax.set_xticklabels(gate_names)
ax.set_yticklabels(gate_names)

# Rotate the tick labels for better readability
plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

# Add text annotations for each cell
for i in range(num_gates):
    for j in range(num_gates):
        # Format in scientific notation
        text = ax.text(j, i, f'{leakage_matrix[i, j]:.1e}',
                      ha="center", va="center", color="white", fontsize=8)

# Labels and title
ax.set_xlabel('Gate electrodes', fontsize=12)
ax.set_ylabel('Gate electrodes', fontsize=12)
ax.set_title('Leakage Test Resistance Matrix', fontsize=14, pad=20)

# Add grid lines between cells
ax.set_xticks(np.arange(num_gates) - 0.5, minor=True)
ax.set_yticks(np.arange(num_gates) - 0.5, minor=True)
ax.grid(which="minor", color="white", linestyle='-', linewidth=2)
ax.tick_params(which="minor", size=0)

plt.tight_layout(pad=2.0)

# Save the figure
output_file = '/Users/rjow/cq/stanza/leakage_test_matrix.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=1.0)
print(f"\nPlot saved to: {output_file}")

# Show the plot
plt.show()
