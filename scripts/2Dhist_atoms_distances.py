# date: Mon May 8, 2023
# @author:Kun-Lin Wu

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


def threshold_percentage(threshold, data):
    list_large = []
    list_small = []
    for i_data in data:
        if i_data > threshold:
            list_large.append(i_data)
        else:
            list_small.append(i_data)
    large_percentage = len(list_large) / len(data)
    small_percentage = len(list_small) / len(data)

    return large_percentage, small_percentage



# GOAL: Get the 2D distance histogram of O-CO2 with Cs-S8R and Cs-S6R in wet Cs/RHO.

# files
# case: wet Cs-RHO (im3m, 6CO2, 15H2O)
# The traj file is the result after a combination of 5 ensembles of different Al distributions.
# data_folder = Path('/Users/calvin11/Library/CloudStorage/Box-Box/project_WetCsRHO_CO2_adsorption/data/02_im3m')
# filepath = data_folder/'im3m_cs10_6co2_15h2o_250.traj'

# case: dry Cs-RHO (i43m, 2CO2).
# The traj file is the result after a combination of 5 ensembles of different Al distributions.
data_folder = Path('/Users/calvin11/Library/CloudStorage/Box-Box/project_WetCsRHO_CO2_adsorption/data/01_i43m')
filepath = data_folder/'i43m_cs10_2co2_250.traj'

# source file: file for identifying potential cation positions in RHO, including d8r, s8r, s4r, center...
source_folder = Path('/Users/calvin11/Library/CloudStorage/Box-Box/project_CO2/src')
# for wet analysis
# file_cation = source_folder/'cation_positions_wet_new.cif'
# for dry analysis
file_cation = source_folder/'cation_positions_dry_new.cif'

# save result
result_folder = Path('/Users/calvin11/Library/CloudStorage/Box-Box/project_WetCsRHO_CO2_adsorption/results/new_0214')


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

# Add the threshold distance of interacting O_CO2 and Cs cations.
# Divide 2D histogram into 4 quadrants.
# Add the percentage values in each quadrant to distinguish the O_CO2 and Cs cations interaction.
threshold = 3.2   # edit  # It is said that the max interacting O-Cs distance is 4 angs.
# data_type_x = xdata   # edit: xdata for vertical line, ydata for horizontal line
# large_percentage_x, small_percentage_x = threshold_percentage(threshold, data_type_x)
# plt.axvline(x=threshold, color='r', linestyle='dashed')   # vertical line
# data_type_y = ydata   # edit: xdata for vertical line, ydata for horizontal line
# large_percentage_y, small_percentage_y = threshold_percentage(threshold, data_type_y)
# plt.axhline(y=threshold, color='r', linestyle='dashed')   # horizontal line
# ratio_bottom_left = small_percentage_x*small_percentage_y*100
# ratio_top_left = small_percentage_x*large_percentage_y*100
# ratio_bottom_right = large_percentage_x*small_percentage_y*100
# ratio_top_right = large_percentage_x*large_percentage_y*100

# add text
# plt.text(2.5, 2.5, 'Both \n {:.2f}'.format(ratio_bottom_left) + '%',
#          color='k', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
# plt.text(2.5, 6.5, '$Cs_{S8R}$'+ '\n {:.2f}'.format(ratio_top_left) + '%',
#          color='k', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
# plt.text(4.5, 2.5, '$Cs_{S6R}$'+ '\n {:.2f}'.format(ratio_bottom_right) + '%',
#          color='k', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
# plt.text(4.5, 6.5, 'None \n {:.2f}'.format(ratio_top_right) + '%',
#          color='k', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

# figure settings
plt.xlabel('$d(O_{CO2}-Cs_{S8R})$, ${\AA}$', size=14)
plt.ylabel('$d(O_{CO2}-Cs_{S6R})$, ${\AA}$', size=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.colorbar()
plt.title('Cs-RHO(i43m), ' + '$2CO_{2}$', size=14)  #$6CO_{2} + 15H_{2}O$
plt.tight_layout()
plt.savefig(result_folder/'2dhist_co2_dualcs_dry.png')
plt.show()



'''
# plot histograms with built in plt.hist
# plt.figure(figsize=(6, 4))
dist = np.concatenate(dist1)
print(dist)
plt.hist(dist, bins=50, density=1, alpha=0.5, color='blue')
plt.ylim(ymin=0.00, ymax=2.0)
plt.xlabel('$d(O_{CO2}-Cs_{S8R})$, ${\AA}$', size=14)
plt.ylabel('Frequency', size=14)
plt.tight_layout()
# plt.savefig('../results/cs_C{CO2}_distance_wet(im3m)_3cases.pdf')
plt.show()
'''





'''
# old
# --get the 2d histogram and 2d contour plot of the closest Cs-d8r or Cs-s6r center distance to the closest Cs-C_CO2
# distance of wet csrho. In addition, calculate the percentage of a group of data in a given threshold value.
# There are two cases in comparison: co2+h2o and CO2 only

# input settings
dict_keys = ['CO2+H2O', 'CO2 only']  # edit: the case
ele1 = 'C_{CO2}'       # edit: the element
ele2 = 'Cs_{d8r}'            # edit: the element
ele3 = 'Cs_{s6r}'          # edit: the element  #'OCO2_{nearCs}','OCO2_{farCs}'

dict_dist1 = {key: [] for key in dict_keys}     # dictionary of ele1-ele2 for two cases
dict_dist2 = {key: [] for key in dict_keys}     # dictionary of ele1-ele3 for two cases
for i_file, filepath in enumerate(all_filepath):
    traj = io.read(filepath, ':')
    # traj = traj_total
    atoms = traj[-1]
    atoms_cation_positions = io.read(file_cation)

    # print('Case analysis: %s' % (dict_keys[i_file]))
    print('Total snapshots analyzed: %s' % (len(traj)))

    # get the closest C_CO2-Cs distance
    dist1 = []  # distance list of ele1-ele2
    dist2 = []  # distance list of ele1-ele3
    for atoms in traj:
        atoms, indices = update_indices(atoms, atoms_cation_positions)  # update atom and indices with the cation file

        # get the closest distance of ele1-ele2 and ele1-ele3
        for i_ele1 in range(len(indices[ele1])):
            # get the distance of ele1 to its corresponding ele2
            if ele1 == 'Cs_{s6r}':
                d = get_cs_s6r_dist_wet(atoms)
            if ele1 == 'Cs_{d8r}':
                d = get_cs_d8r_dist_wet(atoms)
            else:
                dmin = np.min(atoms.get_distances(indices[ele1][i_ele1], indices[ele2], mic=True))
            dist1.append(dmin)
            # dist1.append(d[i_ele1][0])

            # get the closest distance of ele1 and ele3
            d = atoms.get_distances(indices[ele1][i_ele1], indices[ele3], mic=True)     # get the ele1-ele3 distance for each ele1
            dmin = np.min(d)
            dist2.append(dmin)

    # print(dist1)
    print('Total numbers of the closest %s-%s distance data: %s' % (ele1, ele2, len(dist1)))
    # print(dist2)
    print('Total numbers of the closest %s-%s distance data: %s' % (ele1, ele3, len(dist2)))
    avg_dist = sum(dist2)/ len(dist2)
    print('The average distance of $(Cs-C_{CO2})$ is: %s' %(avg_dist))

    dict_dist1[dict_keys[i_file]].append(dist1)
    dict_dist2[dict_keys[i_file]].append(dist2)

print(dict_dist1)
print(dict_dist2)

# -plot 2d histogram
# get the range for the histogram plot
x_range = []
y_range = []
for i_case in range(len(all_filepath)):
    xdata = np.concatenate(dict_dist1[dict_keys[i_case]])
    ydata = np.concatenate(dict_dist2[dict_keys[i_case]])
    x_range.append(np.min(xdata))
    x_range.append(np.max(xdata))
    y_range.append(np.min(ydata))
    y_range.append(np.max(ydata))
hist_range = [[np.min(x_range), np.max(x_range)], [np.min(y_range), np.max(y_range)]]

plt.figure(figsize=(8, 6))
for i_case in range(len(all_filepath)):
    plt.subplot(1, len(all_filepath), i_case + 1)
    xdata = np.concatenate(dict_dist1[dict_keys[i_case]])
    ydata = np.concatenate(dict_dist2[dict_keys[i_case]])
    plt.hist2d(xdata, ydata, bins=(50, 50), range= hist_range, density=True)
    plt.xlabel('$d(%s-%s)$, ${\AA}$' % (ele1, ele2), size=12)
    plt.ylabel('$d(%s-%s)$, ${\AA}$' % (ele1, ele3), size=12)
    plt.colorbar()
    plt.title('%s case' % dict_keys[i_case], size=14)

    # # plot threshold with percentage
    # threshold = 4.0  # edit:threshold
    # data_type = xdata   #edit:xdata for vertical line, ydata for horizontal line
    # large_percentage, small_percentage = threshold_percentage(threshold, data_type)
    # plt.axvline(x=threshold, color='r', linestyle = 'dashed') #vertical line
    # # plt.axhline(y=threshold, color='r', linestyle = 'dashed')   #horizontal line
    # plt.text(3.5, 7.0, '{:.2f}'.format(small_percentage*100) + '%', color='r', fontsize=12)    #edit: text position
    # plt.text(5.5, 7.0, '{:.2f}'.format(large_percentage*100) + '%', color='r', fontsize=12)    #edit: text position

   # plot threshold with percentage
    threshold = 5.0  # edit:threshold
    data_type_x = xdata   #edit:xdata for vertical line, ydata for horizontal line
    large_percentage_x, small_percentage_x = threshold_percentage(threshold, data_type_x)
    plt.axvline(x=threshold, color='r', linestyle = 'dashed') #vertical line
    data_type_y = ydata  # edit:xdata for vertical line, ydata for horizontal line
    large_percentage_y, small_percentage_y = threshold_percentage(threshold, data_type_y)
    plt.axhline(y=threshold, color='r', linestyle = 'dashed')   #horizontal line
    plt.text(4.0, 3.5, '{:.2f}'.format(small_percentage_x*small_percentage_y*100) + '%', color='r', fontsize=12)    #edit: text position
    plt.text(4.0, 7.0, '{:.2f}'.format(small_percentage_x*large_percentage_y*100) + '%', color='r', fontsize=12)    #edit: text position
    plt.text(6.0, 3.5, '{:.2f}'.format(large_percentage_x*small_percentage_y*100) + '%', color='r', fontsize=12)    #edit: text position
    plt.text(6.0, 7.0, '{:.2f}'.format(large_percentage_x*large_percentage_y*100) + '%', color='r', fontsize=12)

plt.suptitle('2D histogram for Cs-RHO (im3m)', size=16)     # edit: title
plt.tight_layout()
# plt.savefig('../results/2dhist_cs_%s_and_cs_%s_wet(im3m).pdf' %(ele2, ele3))
plt.show()

sys.exit()

# -plot 2d histogram contour plot
for i_case in range(len(all_filepath)):
    xdata = np.concatenate(dict_dist1[dict_keys[i_case]])
    ydata = np.concatenate(dict_dist2[dict_keys[i_case]])
    fig = plot_2dhist_contour(xdata,ydata)
    fig.show()

'''