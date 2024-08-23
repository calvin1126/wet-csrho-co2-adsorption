# Import libraries
import sys
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from ase import io
from ase import Atoms
from md_utils import *
from md_tools import *
from md_helper import get_indices, dict_to_atoms

def analyze_distances(traj_file, cation_file, ele1, ele2, ele3):
    """
    Analyze and plot 2D distance histograms for O-CO2 with Cs-S8R and Cs-S6R.

    Parameters:
    traj_file (str): Path to the trajectory file (.traj).
    cation_file (str): Path to the file identifying cation positions (.cif).
    result_file (str): Path to save the resulting 2D histogram.
    ele1 (str): Element 1, e.g., 'C_{CO2}'.
    ele2 (str): Element 2, e.g., 'Cs_{d8r}'.
    ele3 (str): Element 3, e.g., 'Cs_{s6r}'.
    """

    # Read trajectory and cation position files
    traj = io.read(traj_file, ':')
    atoms_cation_positions = io.read(cation_file)
    print(f'Total number of snapshots in the analyzed file: {len(traj)}')

    dist1 = []  # List of O_CO2-ele2 distances
    dist2 = []  # List of O_CO2-ele3 distances

    # Analyze distances
    for atoms in traj:
        atoms, indices = update_indices(atoms, atoms_cation_positions)

        for i_ele in range(len(indices[ele1])):
            # For Cs_{S8R}
            d_ele1_ele2 = atoms.get_distances(indices[ele1][i_ele], indices[ele2], mic=True)
            dmin_ele1_ele2 = np.min(d_ele1_ele2)
            select_index_ele2 = [key for key, val in enumerate(d_ele1_ele2) if val == dmin_ele1_ele2]

            d_o1_ele2 = atoms.get_distances(indices['OCO2_{nearCs}'][i_ele], indices[ele2][select_index_ele2[-1]], mic=True)
            d_o2_ele2 = atoms.get_distances(indices['OCO2_{farCs}'][i_ele], indices[ele2][select_index_ele2[-1]], mic=True)
            dist1.append(min(d_o1_ele2, d_o2_ele2))

            # For Cs_{S6R}
            d_ele1_ele3 = atoms.get_distances(indices[ele1][i_ele], indices[ele3], mic=True)
            dmin_ele1_ele3 = np.min(d_ele1_ele3)
            select_index_ele3 = [key for key, val in enumerate(d_ele1_ele3) if val == dmin_ele1_ele3]

            d_o1_ele3 = atoms.get_distances(indices['OCO2_{nearCs}'][i_ele], indices[ele3][select_index_ele3[-1]], mic=True)
            d_o2_ele3 = atoms.get_distances(indices['OCO2_{farCs}'][i_ele], indices[ele3][select_index_ele3[-1]], mic=True)
            dist2.append(min(d_o1_ele3, d_o2_ele3))

    print(f'Total number of the closest O_CO2-{ele2} distance data: {len(dist1)}')
    print(f'Total number of the closest O_CO2-{ele3} distance data: {len(dist2)}')
    avg_dist = np.mean(dist2)
    print(f'The average distance of (O_CO2-Cs) is: {avg_dist}')

    return dist1, dist2

# Settings for wet and dry cases
case = 'dry'  # Change to 'wet' for wet case

if case == 'wet':
    data_folder = Path('./data/csrho-im3m')
    filepath = data_folder / '6co2_15h2o.traj'
    source_folder = Path('./src')
    file_cation = source_folder / 'cation_positions_wet.cif'
    result_file = Path('./results/2dhist_water_co2_present.pdf')
    ele1 = 'C_{CO2}'
    ele2 = 'Cs_{d8r}'
    ele3 = 'Cs_{s6r}'

elif case == 'dry':
    data_folder = Path('./data/csrho-i43m')
    filepath = data_folder / '2co2.traj'
    source_folder = Path('./src')
    file_cation = source_folder / 'cation_positions_dry.cif'
    result_file = Path('./results/2dhist_co2_cs_dry.pdf')
    ele1 = 'C_{CO2}'
    ele2 = 'Cs_{d8r}'
    ele3 = 'Cs_{s6r}'

# Run the analysis
dist1, dist2 = analyze_distances(filepath, file_cation, ele1, ele2, ele3)
xdata = np.concatenate(dist1)
ydata = np.concatenate(dist2)

# Plot 2D histogram
plt.figure(figsize=(8, 6))
hist_range = [2.1, 7.5]  # Define histogram range
plt.hist2d(xdata, ydata, bins=(50, 50), range=[hist_range, hist_range], density=True)

plt.xlabel(f'$d(O_{{CO2}}-{ele2})$, $\\AA$', size=14)
plt.ylabel(f'$d(O_{{CO2}}-{ele3})$, $\\AA$', size=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.colorbar()
plt.tight_layout()
# plt.savefig(result_file)
plt.show()