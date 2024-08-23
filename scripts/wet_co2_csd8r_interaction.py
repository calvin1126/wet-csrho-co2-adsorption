# import libraries
import sys, os
from pathlib import Path
import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from ase import io
from ase import Atoms
from ase.visualize import view
from md_utils import *
from md_tools import *
from md_helper import get_indices, dict_to_atoms



# GOAL:

# files
# data_folder = Path('./data/csrho-im3m')
# all_filepath = [data_folder/'6co2_15h2o.traj']

# data file for dry Cs/RHO: The traj file is dry Cs/RHO (i43m, 2CO2).
# The traj file is the result after a combination of 5 ensembles of different Al distributions.
data_folder = Path('./data/csrho-i43m')
all_filepath = [data_folder/'2co2.traj']


# source file: file for identifying potential cation positions in RHO, including d8r, s8r, s4r, center...
source_folder = Path('./src')
# file_cation = source_folder/'cation_positions_wet.cif'
file_cation = source_folder/'cation_positions_dry.cif'

# save result
result_folder = Path('./results')


# input settings
# dict_keys = ['$6CO_{2}+15H_{2}O$']
dict_keys = ['$2CO_{2}$']
dict_dist = {key: [] for key in dict_keys}
ele1 = 'C_{CO2}'    # edit: the element
ele2 = 'Cs_{d8r}'   # edit: the element  # Cs_{d8r} is referring to Cs cations at S8R site.

for i_file, filepath in enumerate(all_filepath):
    traj = io.read(filepath, ':')   # read the traj file
    atoms_cation_positions = io.read(file_cation)  # read the file with specific position for cations
    print('Total number of snapshots in the analyzed file: %s' % (len(traj)))


    dist1 = []   # list of O1_CO2-ele2 distance
    for atoms in traj:
        atoms, indices = update_indices(atoms, atoms_cation_positions)  # update atom and indices with the cation file

        # For each CO2, identify the possible interacting Cs cations based on distances.
        # First, get the closest distance of C_CO2 to Cs cations.
        # Second, identify the two O atoms beside C_CO2 and calculate their distances with the closest Cs cation.
        # Third, select the smaller distance and append to the list of O_CO2-Cs distance.
        for i_ele in range(len(indices[ele1])):
            # For Cs_{S8R}
            # get the closest distance of C_CO2 to Cs cations
            d_ele1_ele2 = atoms.get_distances(indices[ele1][i_ele], indices[ele2], mic=True)
            dmin_ele1_ele2 = np.min(d_ele1_ele2)
            select_index_ele2 = [key for key, val in enumerate(d_ele1_ele2) if val == dmin_ele1_ele2]

            # identify the two O atoms beside C_CO2 and calculate their distances with the closest Cs cation.
            # select the smaller distance and append to the list of O_CO2-Cs distance.
            d_o1_ele2 = atoms.get_distances(indices['OCO2_{nearCs}'][i_ele], indices[ele2][select_index_ele2[-1]], mic=True)
            d_o2_ele2 = atoms.get_distances(indices['OCO2_{farCs}'][i_ele], indices[ele2][select_index_ele2[-1]], mic=True)
            if d_o1_ele2 < d_o2_ele2:
                dist1.append(d_o1_ele2)
            else:
                dist1.append(d_o2_ele2)

    dict_dist[dict_keys[i_file]].append(np.concatenate(dist1))

print(dict_dist)

# plot histograms with built in plt.hist
plt.figure(figsize=(6, 2))
for i_case in range(len(all_filepath)):
    plt.subplot(len(all_filepath), 1, i_case+1)
    data = np.concatenate(dict_dist[dict_keys[i_case]])
    plt.hist(data, bins=50, density=1, alpha=0.5, label=dict_keys[i_case], color='orange')
    plt.legend()
    plt.axvspan(xmin=2.00, xmax=3.60, color='green', alpha=0.2)
    # plt.xlim(xmin=2.00, xmax=7.50)
    plt.xlim(xmin=2.00, xmax=7.50)
    plt.ylim(ymin=0.00, ymax=2.00)
    plt.ylabel('Frequency', size=14)
    plt.text(2.5, 1.5, 'Interacting \n region', verticalalignment='center',
             color='k', fontsize=10)
    plt.text(4.5, 1.5, 'Non-interacting \n region', verticalalignment='center',
             color='k', fontsize=10)

plt.xlabel('$d(O_{CO2}-Cs_{S8R})$, ${\AA}$', size=14)

plt.tight_layout()
plt.savefig(result_folder/'hist_co2_cs_dry.pdf')
plt.show()



