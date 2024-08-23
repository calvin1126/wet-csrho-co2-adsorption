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


# files
data_folder = Path('./data/csrho-i43m')
# data file for wet Cs/RHO: The traj file is wet Cs/RHO (im3m, 6CO2, 15H2O).
# The traj file is the result after a combination of 5 ensembles of different Al distributions.
all_filepath = [data_folder/'2co2.traj']
# all_filepath = [data_folder/'6co2.traj']

# source file: file for identifying potential cation positions in RHO, including d8r, s8r, s4r, center...
source_folder = Path('./src')
# for dry analysis
file_cation = source_folder/'cation_positions_dry.cif'

# save result
result_folder = Path('./results')



#--get the cs-d8r center distance (only axial) histogram of wet csrho in comparison of three cases: co2+h2o,
# co2 only and h2o only

#input settings
dict_keys = ['$2CO_{2}$']  #the name can be changed
# dict_keys = ['$6CO_{2}$']
ele1 = 'Cs_{d8r}'
dict_dist = {key: [] for key in dict_keys}

for i_file, filepath in enumerate(all_filepath):
    traj = io.read(filepath,':')
    atoms_cation_positions = io.read(file_cation)
    atoms = traj[-1]
    atoms, indices = update_indices(atoms, atoms_cation_positions)
    print('Case analysis: %s' % (dict_keys[i_file]))
    print('Total snapshots analyzed %s' % len(traj))

    dict_distance_vector = get_dict_distance_vector(ele1, indices)  # distance list of ele1-ele2 by vector

    for atoms in traj:
        atoms, indices = update_indices(atoms, atoms_cation_positions)  # update atom and indices with the cation file
        # get the distance of Cs_{d8r} to its corresponding d8r
        d = get_cs_d8r_dist_dry_vector(atoms, indices)
        get_distance_by_vector(ele1, indices, d, dict_distance_vector)

    # get the list of axial distance of Cs_{d8r} to its corresponding d8r
    list_axial, list_lateral = get_cs_d8r_axial_lateral_dist_list_wet(indices, dict_distance_vector)
    data_axial = np.concatenate(list_axial)  # remove the outside bracket
    data_lateral = np.concatenate(list_lateral)
    print('Total numbers of the closest %s-d8r distance data: %s' % (ele1, len(data_axial)))
    dict_dist[dict_keys[i_file]].append(data_lateral)

print(dict_dist)

# -plot
plt.figure(figsize=(6, 2))

for i_case in range(len(all_filepath)):
    plt.subplot(len(all_filepath), 1, i_case+1)
    abs_axial = [abs(ele) for ele in dict_dist[dict_keys[i_case]]]
    plt.hist(abs_axial, bins=150, density=1, alpha=0.5, label=dict_keys[i_case], color='orange')
    plt.xlim(xmin=0.00, xmax=3.00)
    # plt.ylim(ymin=0.00, ymax=3.0)
    plt.ylabel('Frequency')
    plt.legend()

plt.xlabel('$d(%s-%s)$, ${\AA}$' % ('Cs_{D8R}', 'D8R'))
# plt.suptitle('$(Cs_{D8r}-D8R_{center})$ lateral displacement in Cs/RHO (i43m)')
plt.tight_layout()
# plt.savefig(result_folder/'cs_d8r_axial_displacement_dry(i43m)_2CO2.pdf')
plt.show()







'''
#--get the closest Cs-C_CO2 distance histogram of wet csrho in comparison of three cases: co2+h2o, co2 only and h2o only
dict_keys = ['Dry']  #the name can be changed
dict_dist = {key: [] for key in dict_keys}
ele1 = 'C_{CO2}'
ele2 = 'Cs_{d8r}'   #'Cs_{d8r}'

for i_file, filepath in enumerate(all_filepath):
    traj = io.read(filepath,':')
    atoms_cation_positions = io.read(file_cation)
    atoms = traj[-1]
    atoms, indices = update_indices(atoms, atoms_cation_positions)
    print('Total snapshots analyzed %s' % len(traj))

    #for identifying each component of a vector
    list_dist = []
    for atoms in traj:
        atoms, indices = update_indices(atoms, atoms_cation_positions)

        # get the CO2 to its corresponding Cs_{d8r} in Cs-RHO
        for i_ele in range(len(indices[ele1])):
            d = atoms.get_distances(indices[ele1][i_ele], indices[ele2][i_ele], mic=True)
            list_dist.append(d[-1])

        # if (bool(indices.get(ele1)) == True) or (bool(indices.get(ele2)) == True):
        # # get the min of ele1 - ele2 distance
        #     for i_ele in range(len(indices[ele1])):
        #         d = np.min(atoms.get_distances(indices[ele1][i_ele], indices[ele2], mic=True))
        #         list_dist.append(d)

    # print(list_dist)
    print('Total numbers of the closest %s-%s distance data: %s' %(ele1, ele2, len(list_dist)))

    dict_dist[dict_keys[i_file]].append(list_dist)

# print([i for i, j in enumerate(list_dist) if j == 0.0])
# sys.exit()
print(dict_dist)

# plot histograms with built in plt.hist and best fit line from scipy.norm
plt.figure(figsize=(6, 4))

for i_case in range(len(all_filepath)):
    plt.subplot(len(all_filepath), 1, i_case+1)
    dist = [ele for ele in dict_dist[dict_keys[i_case]]]
    plt.hist(dist, bins=50, density=1, alpha=0.5, label=dict_keys[i_case], color='blue')

    # # add a 'best fit' line
    # if (bool(dist[0]) ==True):
    #     mean, var = norm.fit(dist)
    #     x = np.linspace(0.00, 9.00, len(dist[0]))  #edit: range
    #     p = norm.pdf(x, mean, var)
    #     plt.plot(x, p, 'k', linewidth=1)
    #     plt.title('Fit: mean= {:.2f} and std= {:.2f}'.format(mean, var))

    # # plot threshold with percentage
    # threshold = 1.50  # edit:threshold
    # data_type = dist[0]  # edit:data_type
    # large_percentage, small_percentage = threshold_percentage(threshold, data_type)
    # plt.axvline(x=threshold, color='r', linestyle = 'dashed') #vertical line
    # plt.text(1.0, 2.5, '{:.2f}'.format(small_percentage * 100) + '%', color='r', fontsize=12)  # edit: text position
    # plt.text(5.0, 2.5, '{:.2f}'.format(large_percentage * 100) + '%', color='r', fontsize=12)  # edit: text position

    plt.xlim(xmin=0.00)  #edit: range
    plt.ylim(ymin=0.00, ymax=3.0)   #edit: range
    # plt.title('threshold= {:.2f}'.format(threshold))
    plt.ylabel('Frequency')
    plt.legend()

plt.xlabel('$d(%s-%s)$, ${\AA}$' % (ele1, ele2))
plt.suptitle('$(%s-%s})$ displacement for Cs-RHO (i43m, 2 CO2)' % (ele1, ele2))
plt.tight_layout()
# plt.savefig('../results/cs_C{CO2}_distance_wet(im3m)_3cases.pdf')
plt.show()





sys.exit()




#--get the cs-d8r center distance histogram of csrho in comparison of both axial and lateral directions

#input settings
case_name = ['dry']  #the name can be changed
ele1 = 'Cs_{d8r}'

for i_file, filepath in enumerate(all_filepath):
    traj = io.read(filepath, ':')
    atoms_cation_positions = io.read(file_cation)
    atoms = traj[-1]
    atoms, indices = update_indices(atoms, atoms_cation_positions)
    print('Case analysis: %s' % (case_name[i_file]))
    print('Total snapshots analyzed %s' % len(traj))

    dict_distance_vector = get_dict_distance_vector(ele1, indices)

    for atoms in traj:
        atoms, indices = update_indices(atoms, atoms_cation_positions)  # update atom and indices with the cation file
        # get the distance of Cs_{d8r} to its corresponding d8r
        d = get_cs_d8r_dist_dry_vector(atoms)
        get_distance_by_vector(ele1, dict_distance_vector)

    list_axial, list_lateral = get_cs_d8r_axial_lateral_dist_list_wet(indices, dict_distance_vector)
    data_axial = np.concatenate(list_axial)  #remove the outside bracket
    data_lateral = np.concatenate(list_lateral)  #remove the outside bracket

    # -plot
    # this is to reflect the negative value
    abs_ydata_axial = [abs(ele) for ele in data_axial]
    abs_ydata_lateral = [abs(ele) for ele in data_lateral]

    # # best fit of data
    # mu, std = norm.fit(data_axial)
    # mu_l, std_l = norm.fit(data_lateral)

    plt.subplot(2, 1, 1)
    plt.hist(abs_ydata_axial, bins=50, density=1, alpha=0.5, label='axial_dir', color='blue')
    plt.xlim(xmin=0.00, xmax=3.00)
    plt.ylim(ymin=0.00, ymax=3.0)
    plt.ylabel('Frequency')
    plt.legend()
    # # add a 'best fit' line
    # x = np.linspace(-4.00, 4.00, 100)
    # p = norm.pdf(x, mu, std)
    # plt.plot(x, p, 'k', linewidth=1)
    # plt.title('Fit Values: mean- {:.2f} and std- {:.2f}'.format(mu, std))

    plt.subplot(2, 1, 2)
    plt.hist(abs_ydata_lateral, bins=50, density=1, alpha=0.5, label='lateral_dir', color='orange')
    plt.xlim(xmin=0.00, xmax=3.00)
    plt.ylim(ymin=0.00, ymax=3.0)
    plt.ylabel('Frequency')
    plt.legend()
    # # add a 'best fit' line
    # x = np.linspace(-4.00, 4.00, 100)
    # p = norm.pdf(x, mu_l, std_l)
    # plt.plot(x, p, 'k', linewidth=1)
    # plt.title('Fit Values: mean- {:.2f} and std- {:.2f}'.format(mu_l, std_l))

    plt.xlabel('$d(%s-%s)$, ${\AA}$' % ('Cs_{d8r}', 'd8r'))
    plt.suptitle('$(Cs_{d8r}-d8r_{center})$ displacement for Cs-RHO (i43m) for %s case' % case_name[i_file])
    plt.tight_layout()
    plt.savefig('/Users/calvin11/Library/CloudStorage/Box-Box/project_CO2/results/001_csrho_i43mdry_2co2/cs_d8r_axial_lateral_dry(i43m).pdf')
    plt.show()

'''
