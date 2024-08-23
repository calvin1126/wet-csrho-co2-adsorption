# Import required libraries
import sys, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from ase import io
from ase import Atoms
from ase.visualize import view
from md_utils import *
from md_tools import *
from md_helper import *
from md_dry_csrho_functions import *

# Define paths to data, source files, and results
data_folder = Path('./data/csrho-i43m')
source_folder = Path('./src')
result_folder = Path('./results')

# Define the trajectory files for dry Cs/RHO: i43m phase with 2CO2
trajectory_files = [
    data_folder / '2co2.traj'
]

# Define the source file for identifying potential cation positions in RHO
file_cation_positions = source_folder / 'cation_positions_dry.cif'

# Define the case for analysis (for reproducibility, key can be modified)
case_labels = ['$2CO_{2}$']

# Initialize a dictionary to store lateral distances for the case
element_label = 'Cs_{d8r}'
distance_data = {label: [] for label in case_labels}

# Process each trajectory file
for idx, filepath in enumerate(trajectory_files):
    # Read trajectory data and the cation positions file
    trajectory = io.read(filepath, ':')
    cation_positions = io.read(file_cation_positions)
    final_frame = trajectory[-1]

    # Update indices based on cation positions
    updated_atoms, indices = update_indices(final_frame, cation_positions)
    print(f'Analyzing case: {case_labels[idx]}')
    print(f'Total snapshots analyzed: {len(trajectory)}')

    # Get the distance vector dictionary for Cs-d8r distances
    distance_vectors = get_dict_distance_vector(element_label, indices)

    # Iterate through each frame in the trajectory
    for atoms in trajectory:
        # Update atoms and indices with the cation positions
        atoms, indices = update_indices(atoms, cation_positions)
        # Calculate Cs_{d8r}-d8r distances for the dry system
        distances = get_cs_d8r_dist_dry_vector(atoms, indices)
        get_distance_by_vector(element_label, indices, distances, distance_vectors)

    # Extract axial and lateral distances for Cs_{d8r}-d8r
    axial_distances, lateral_distances = get_cs_d8r_axial_lateral_dist_list_wet(indices, distance_vectors)
    distance_data[case_labels[idx]].append(np.concatenate(lateral_distances))  # Flatten list of lateral distances

    print(f'Total {element_label}-d8r distance data points: {len(np.concatenate(axial_distances))}')

# Plot the lateral displacement histogram for the case
plt.figure(figsize=(6, 2))

for i, label in enumerate(case_labels):
    plt.subplot(len(case_labels), 1, i + 1)
    absolute_lateral_distances = [abs(dist) for dist in distance_data[label]]
    plt.hist(absolute_lateral_distances, bins=150, density=True, alpha=0.5, label=label, color='orange')
    plt.xlim(0.00, 3.00)
    plt.ylabel('Frequency')
    plt.legend()

plt.xlabel(f'$d({element_label}-D8R)$,' + ' ${\AA}$')
plt.tight_layout()
# plt.savefig(result_folder/'cs_d8r_axial_displacement_i43m.pdf')
plt.show()
