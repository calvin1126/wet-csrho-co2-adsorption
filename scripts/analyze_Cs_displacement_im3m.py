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
from md_wet_csrho_functions import *

# Set up paths to data, source files, and results
data_folder = Path('./data/csrho-im3m')
source_folder = Path('./src')
result_folder = Path('./results')

# Define the trajectory files for wet Cs/RHO: im3m phase with CO2 and H2O
trajectory_files = [
    data_folder / '6co2_15h2o.traj',
    data_folder / '15h2o.traj'
]

# Define the source file for identifying potential cation positions in RHO
file_cation_positions = source_folder / 'cation_positions_wet.cif'

# Define the cases for analysis (for reproducibility, keys can be modified)
case_labels = ['$6CO_{2}+15H_{2}O$', '$15H_{2}O$']

# Initialize a dictionary to store axial distances for each case
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
        # Calculate Cs_{d8r}-d8r distances
        distances = get_cs_d8r_dist_wet_vector(atoms, indices)
        get_distance_by_vector(element_label, indices, distances, distance_vectors)

    # Extract axial and lateral distances for Cs_{d8r}-d8r
    axial_distances, lateral_distances = get_cs_d8r_axial_lateral_dist_list_wet(indices, distance_vectors)
    distance_data[case_labels[idx]].append(np.concatenate(axial_distances))  # Flatten list of axial distances

    print(f'Total {element_label}-d8r distance data points: {len(np.concatenate(axial_distances))}')

# Plot the axial displacement histograms for each case
plt.figure(figsize=(6, 4))

for i, label in enumerate(case_labels):
    plt.subplot(len(case_labels), 1, i+1)
    absolute_axial_distances = [abs(dist) for dist in distance_data[label]]
    plt.hist(absolute_axial_distances, bins=50, density=True, alpha=0.5, label=label, color='blue')
    plt.xlim(0.00, 3.00)
    plt.ylim(0.00, 3.00)
    plt.ylabel('Frequency')
    plt.legend()

plt.xlabel(f'$d({element_label}-D8R)$,' + ' ${\AA}$')
plt.tight_layout()
plt.savefig(result_folder/'cs_d8r_axial_displacement_im3m.pdf')
plt.show()
