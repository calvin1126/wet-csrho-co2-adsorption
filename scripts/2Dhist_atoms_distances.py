# import libraries
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
from md_helper import get_indices, dict_to_atoms



# GOAL: Get the 2D distance histogram of O-CO2 with Cs-S8R and Cs-S6R in wet Cs/RHO.

# files
# case: wet Cs-RHO (im3m, 6CO2, 15H2O)
# The traj file is the result after a combination of 5 ensembles of different Al distributions.
# data_folder = Path('./data/csrho-im3m')
# filepath = data_folder/'6co2_15h2o.traj'

# case: dry Cs-RHO (i43m, 2CO2).
# The traj file is the result after a combination of 5 ensembles of different Al distributions.
data_folder = Path('./data/csrho-i43m')
filepath = data_folder/'2co2.traj'

# source file: file for identifying potential cation positions in RHO, including d8r, s8r, s4r, center...
source_folder = Path('./src')
# for wet analysis
# file_cation = source_folder/'cation_positions_wet.cif'
# for dry analysis
file_cation = source_folder/'cation_positions_dry.cif'

# save result
result_folder = Path('./results')


# input settings
ele1 = 'C_{CO2}'    # edit: the element
ele2 = 'Cs_{d8r}'   # edit: the element  # Cs_{d8r} is referring to Cs cations at S8R site.
ele3 = 'Cs_{s6r}'   # edit: the element


# main
traj = io.read(filepath, ':')   # read the traj file
atoms_cation_positions = io.read(file_cation)  # read the file with specific position for cations
print('Total number of snapshots in the analyzed file: %s' % (len(traj)))


dist1 = []   # list of O1_CO2-ele2 distance
dist2 = []   # list of O2_CO2-ele3 distance
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

        # For Cs_{S6R}
        # get the closest distance of C_CO2 to Cs cations
        d_ele1_ele3 = atoms.get_distances(indices[ele1][i_ele], indices[ele3], mic=True)
        dmin_ele1_ele3 = np.min(d_ele1_ele3)
        select_index_ele3 = [key for key, val in enumerate(d_ele1_ele3) if val == dmin_ele1_ele3]

        # identify the two O atoms beside C_CO2 and calculate their distances with the closest Cs cation.
        # select the smaller distance and append to the list of O_CO2-Cs distance.
        d_o1_ele3 = atoms.get_distances(indices['OCO2_{nearCs}'][i_ele], indices[ele3][select_index_ele3[-1]], mic=True)
        # check the second O atom in CO2 that is within the interacting distance range with Cs
        d_o2_ele3 = atoms.get_distances(indices['OCO2_{farCs}'][i_ele], indices[ele3][select_index_ele3[-1]], mic=True)
        if d_o1_ele3 < d_o2_ele3:
            dist2.append(d_o1_ele3)
        else:
            dist2.append(d_o2_ele3)


print('Total numbers of the closest O_{CO2}-%s distance data: %s' % (ele2, len(dist1)))
print('Total numbers of the closest O_{CO2}-%s distance data: %s' % (ele3, len(dist2)))

avg_dist = sum(dist2)/ len(dist2)
print('The average distance of $(O_{CO2}-Cs)$ is: %s' %(avg_dist))


# plot 2d histogram of O-CO2 with Cs-S8R and Cs-S6R
plt.figure(figsize=(8, 6))

# get the data
xdata = np.concatenate(dist1)
ydata = np.concatenate(dist2)

# define histogram range
hist_range = [2.1, 7.5]   # edit

# plot
plt.hist2d(xdata, ydata, bins=(50, 50), range=[hist_range, hist_range], density=True)

# figure settings
plt.xlabel('$d(O_{CO2}-Cs_{S8R})$, ${\AA}$', size=14)
plt.ylabel('$d(O_{CO2}-Cs_{S6R})$, ${\AA}$', size=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.colorbar()
plt.title('$2CO_{2}$', size=14) 
plt.tight_layout()
plt.savefig(result_folder/'2dhist_co2_cs_dry.pdf')
plt.show()
