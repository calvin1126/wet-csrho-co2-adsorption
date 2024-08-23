# Import necessary libraries
import sys, os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from ase import io
from ase import Atoms
from md_utils import *
from md_tools import *
from md_helper import get_indices, dict_to_atoms

# Set up paths for data and source files
case = 'wet'  # Change to 'wet' for wet case

if case == 'wet':
    data_folder = Path('./data/csrho-im3m')
    all_filepath = [data_folder / '6co2_15h2o.traj']
    source_folder = Path('./src')
    file_cation = source_folder / 'cation_positions_wet.cif'
    case_color = 'blue'
elif case == 'dry':
    data_folder = Path('./data/csrho-i43m')
    all_filepath = [data_folder/'2co2.traj']  # Filepath for the dry Cs/RHO system with 2CO2
    source_folder = Path('./src')
    file_cation = source_folder/'cation_positions_dry.cif'  # File for identifying potential cation positions in RHO
    case_color = 'orange'

result_folder = Path('./results')  # Folder to save the results

# Input settings
dict_keys = ['$2CO_{2}$']  # Key for labeling the dataset in the plots
dict_dist = {key: [] for key in dict_keys}  # Initialize a dictionary to store distances
ele1 = 'C_{CO2}'    # Element: Carbon in CO2
ele2 = 'Cs_{d8r}'   # Element: Cesium at the S8R site

# Loop through each trajectory file
for i_file, filepath in enumerate(all_filepath):
    traj = io.read(filepath, ':')  # Read the trajectory file
    atoms_cation_positions = io.read(file_cation)  # Read the file with specific cation positions
    print(f'Total number of snapshots in the analyzed file: {len(traj)}')

    dist1 = []  # List to store distances between O_CO2 and Cs

    for atoms in traj:
        # Update atom and indices with the cation positions
        atoms, indices = update_indices(atoms, atoms_cation_positions)

        # For each CO2, identify the possible interacting Cs cations based on distances
        for i_ele in range(len(indices[ele1])):
            # Get the closest distance of C_CO2 to Cs cations
            d_ele1_ele2 = atoms.get_distances(indices[ele1][i_ele], indices[ele2], mic=True)
            dmin_ele1_ele2 = np.min(d_ele1_ele2)
            select_index_ele2 = [key for key, val in enumerate(d_ele1_ele2) if val == dmin_ele1_ele2]

            # Calculate distances of O_CO2 atoms with the closest Cs cation
            d_o1_ele2 = atoms.get_distances(indices['OCO2_{nearCs}'][i_ele], indices[ele2][select_index_ele2[-1]], mic=True)
            d_o2_ele2 = atoms.get_distances(indices['OCO2_{farCs}'][i_ele], indices[ele2][select_index_ele2[-1]], mic=True)

            # Append the smaller distance to the list
            dist1.append(min(d_o1_ele2, d_o2_ele2))

    # Store the distances in the dictionary
    dict_dist[dict_keys[i_file]].append(np.concatenate(dist1))

# Plot histograms of the distances
plt.figure(figsize=(6, 2))
for i_case in range(len(all_filepath)):
    plt.subplot(len(all_filepath), 1, i_case + 1)
    data = np.concatenate(dict_dist[dict_keys[i_case]])
    plt.hist(data, bins=50, density=1, alpha=0.5, label=dict_keys[i_case], color= case_color)
    plt.legend()
    plt.axvspan(xmin=2.00, xmax=3.60, color='green', alpha=0.2)
    plt.xlim(xmin=2.00, xmax=7.50)
    plt.ylim(ymin=0.00, ymax=2.00)
    plt.ylabel('Frequency', size=14)
    plt.text(2.5, 1.5, 'Interacting \n region', verticalalignment='center', color='k', fontsize=10)
    plt.text(4.5, 1.5, 'Non-interacting \n region', verticalalignment='center', color='k', fontsize=10)

plt.xlabel('$d(O_{CO2}-Cs_{S8R})$, ${\AA}$', size=14)
plt.tight_layout()
# plt.savefig(result_folder/'hist_co2_cs_dry.pdf')
plt.show()
