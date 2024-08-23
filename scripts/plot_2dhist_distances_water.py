# Import necessary libraries
import sys, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from ase import io
from ase.visualize import view
from md_utils import *
from md_tools import *
from md_helper import get_indices, dict_to_atoms

# Goal: Analyze the 2D distance histogram between Cs_{d8r}, d8r, and O_{H2O} in a wet Cs/RHO system.

# File paths and directories
data_folder = Path('./data/csrho-im3m')
trajectory_file = data_folder / '6co2_15h2o.traj'
source_folder = Path('./src')
cation_positions_file = source_folder / 'cation_positions_wet.cif'
result_folder = Path('./results/')

# Define elements of interest
ele1 = 'Cs_{d8r}'  # Cesium at d8r positions
ele2 = 'd8r'       # d8r positions
ele3 = 'O_{H2O}'   # Oxygen from water

# Load trajectory data and cation positions
trajectory = io.read(trajectory_file, ':')
cation_positions = io.read(cation_positions_file)
print(f'Total number of snapshots in the analyzed file: {len(trajectory)}')

# Initialize lists to store distances
distances_ele1_ele2 = []  # Cs_{d8r} to d8r distances
distances_ele1_ele3 = []  # Cs_{d8r} to O_{H2O} distances

# Calculate distances for each snapshot
for atoms in trajectory:
    atoms, indices = update_indices(atoms, cation_positions)
    
    for i_ele in range(len(indices[ele1])):
        # Minimum distance between Cs_{d8r} and d8r
        dmin_ele1_ele2 = np.min(atoms.get_distances(indices[ele1][i_ele], indices[ele2], mic=True))
        distances_ele1_ele2.append(dmin_ele1_ele2)
        
        # Minimum distance between Cs_{d8r} and O_{H2O}
        dmin_ele1_ele3 = np.min(atoms.get_distances(indices[ele1][i_ele], indices[ele3], mic=True))
        distances_ele1_ele3.append(dmin_ele1_ele3)

print(f'Total number of closest {ele1}-{ele2} distance data: {len(distances_ele1_ele2)}')
print(f'Total number of closest {ele1}-{ele3} distance data: {len(distances_ele1_ele3)}')

# Plot 2D histogram of the distances
plt.figure(figsize=(8, 6))
plt.hist2d(distances_ele1_ele2, distances_ele1_ele3, bins=(50, 50), density=True)

# Set plot labels and titles
plt.xlabel(f'$d({ele1}-{ele2})$, $\AA$', size=14)
plt.ylabel(f'$d({ele1}-{ele3})$, $\AA$', size=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.colorbar(label='Density')
plt.title('$6CO_{2} + 15H_{2}O$', size=14)
plt.tight_layout()

# Save the plot
output_file = result_folder / '2dhist_water_co2_present.pdf'
plt.savefig(output_file)
plt.show()
