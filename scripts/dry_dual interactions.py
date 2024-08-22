# date: Oct 25, 2023
# @author:Kun-Lin Wu

import sys, os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from ase import io
from ase import Atoms
from ase.visualize import view
from md_utils import *
from md_tools import *
from md_helper import get_indices, dict_to_atoms


# files
data_folder = Path('/Users/calvin11/Library/CloudStorage/Box-Box/project_CO2/data/paper_cs-rho')
# data file for dry Cs/RHO: The traj file is dry Cs/RHO (i43m, 2CO2).
# The traj file is the result after a combination of 5 ensembles of different Al distributions.
filepath = data_folder/'i43m_cs10_2co2_250.traj'

# source file: file for identifying potential cation positions in RHO, including d8r, s8r, s4r, center...
source_folder = Path('/Users/calvin11/Library/CloudStorage/Box-Box/project_CO2/src')
# for dry analysis
file_cation = source_folder/'cation_positions_dry_new.cif'

# save result
result_folder = Path('/Users/calvin11/Library/CloudStorage/Box-Box/project_WetCsRHO_CO2_adsorption/results/paper_csrho')


# input settings
ele1 = 'C_{CO2}'   # edit: the element
ele2 = 'Cs'        # edit: the element


#--main
traj = io.read(filepath, ':')
atoms_cation_positions = io.read(file_cation)
atoms = traj[0]
atoms, indices = update_indices(atoms, atoms_cation_positions)
print('Total snapshots analyzed %s' % len(traj))

dict_all_cs = []
for i_ele in range(len(indices[ele1])):
    num_cs_total = []
    num_cs_d8r = []
    num_cs_s6r = []
    wrong_index = []
    for i, atoms in enumerate(traj):
        atoms, indices = update_indices(atoms, atoms_cation_positions)
        print(indices)

        # check the first O atom in CO2 that is within the interacting distance range with Cs
        dist_o1_cs = atoms.get_distances(indices['O_{CO2}'][i_ele], indices[ele2], mic=True)  # !!!need to renaming the OCO2
        dist_select_o1 = [i_dist for i_dist in dist_o1_cs if i_dist < 3.7]
        dist_select_index_o1 = [key for key, val in enumerate(dist_o1_cs) if val in set(dist_select_o1)]
        # print('o1_cs:', dist_select_o1)
        # print('o1_cs_index:', dist_select_index_o1)

        # check the second O atom in CO2 that is within the interacting distance range with Cs
        # dist_o2_cs = atoms.get_distances(indices['OCO2_2'][i_ele], indices[ele2],
        #                                  mic=True)  # !!!need to renaming the OCO2
        # dist_select_o2 = [i_dist for i_dist in dist_o2_cs if i_dist < 4.0]
        # dist_select_index_o2 = [key for key, val in enumerate(dist_o2_cs) if val in set(dist_select_o2)]
        # print('o2_cs:', dist_select_o2)
        # print('o2_cs_index:', dist_select_index_o2)

        o_cs_index_combine = dist_select_index_o1 #+ dist_select_index_o2
        o_cs_index = [*set(o_cs_index_combine)]  # this is to prevent duplicate numbers
        print('The index of Cs that interacts with CO2: ', o_cs_index)


        # classify the cs that co2 is interacting with
        target_cs_indices = [val for key, val in enumerate(indices[ele2]) if key in set(o_cs_index)]
        print(target_cs_indices)
        num_cs_total.append(len(target_cs_indices))
        cs_d8r_co2 = [val for key, val in enumerate(indices['Cs_{d8r}']) if val in set(target_cs_indices)]
        cs_s6r_co2 = [val for key, val in enumerate(indices['Cs_{s6r}']) if val in set(target_cs_indices)]
        print('The number of Cs atoms that binds with CO2', len(target_cs_indices))
        print('The Cs near d8r center that binds with CO2:', cs_d8r_co2)
        print('The number of Cs-d8r that binds with CO2:', len(cs_d8r_co2))
        num_cs_d8r.append(len(cs_d8r_co2))
        print('The Cs near s6r center that binds with CO2:', cs_s6r_co2)
        print('The number of Cs-s6r that binds with CO2:', len(cs_s6r_co2))
        num_cs_s6r.append(len(cs_s6r_co2))


    print('Number of Cs cations interacting with CO2:', num_cs_total)
    print('Number of Cs(S8R) interacting with CO2:   ', num_cs_d8r)
    print('Number of Cs(S6R) interacting with CO2:   ', num_cs_s6r)



    # get the dictionary of the numbers and types of Cs cations interacting with CO2
    dict_num_cs = {}
    for i_num_cs in range(4):
        dict_num_cs['num_cs_%s' % (i_num_cs)] = num_cs_total.count(i_num_cs)
    print(dict_num_cs)

    # discussion of number of cs_d8r and cs_s6r
    dict_types_cs = {}

    # identify which Cs(S8R/ S6r) when there is total "1 Cs" interacting with CO2
    list_cs_1_d8r = [num_cs_d8r[key] for key, val in enumerate(num_cs_total) if val == 1]
    list_cs_1_s6r = [num_cs_s6r[key] for key, val in enumerate(num_cs_total) if val == 1]
    dict_types_cs['num_cs_1_d8r'] = list_cs_1_d8r.count(1)
    dict_types_cs['num_cs_1_s6r'] = list_cs_1_s6r.count(1)

    # identify which Cs(S8R/ S6r) when there is total "2 Cs" interacting with CO2
    list_cs_2_d8r = [num_cs_d8r[key] for key, val in enumerate(num_cs_total) if val == 2]
    list_cs_2_s6r = [num_cs_s6r[key] for key, val in enumerate(num_cs_total) if val == 2]
    assert len(list_cs_2_d8r) == len(list_cs_2_s6r)
    dict_types_cs['num_cs_2_d8r'] = list_cs_2_d8r.count(2)
    dict_types_cs['num_cs_2_s6r'] = list_cs_2_s6r.count(2)
    dict_types_cs['num_cs_1_d8r_1_s6r'] = sum(list_cs_2_d8r[i] == 1 for i in range(len(list_cs_2_d8r)))

    # identify which Cs(S8R/ S6r) when there is total "3 Cs" interacting with CO2
    list_cs_3_d8r = [num_cs_d8r[key] for key, val in enumerate(num_cs_total) if val == 3]
    list_cs_3_s6r = [num_cs_s6r[key] for key, val in enumerate(num_cs_total) if val == 3]
    assert len(list_cs_3_d8r) == len(list_cs_3_s6r)
    dict_types_cs['num_cs_3_d8r'] = list_cs_3_d8r.count(3)
    dict_types_cs['num_cs_3_s6r'] = list_cs_3_s6r.count(3)
    dict_types_cs['num_cs_1_d8r_2_s6r'] = sum(list_cs_3_d8r[i] == 1 for i in range(len(list_cs_3_d8r)))
    dict_types_cs['num_cs_2_d8r_1_s6r'] = sum(list_cs_3_d8r[i] == 2 for i in range(len(list_cs_3_d8r)))
    print(dict_types_cs)


    # plot
    # plot_pie_num_types_cs(i_ele, dict_num_cs, dict_types_cs)

    # preparing datat for overall analysis by appending each dictionary
    dict_all_cs.append(dict_num_cs)
    dict_all_cs.append(dict_types_cs)

# get the dictionary containing the total numbers and types of cations interacting with CO2
dict_all_result = {k: sum(d[k] for d in dict_all_cs if k in d) for k in set(k for d in dict_all_cs for k in d)}

print(dict_all_result)
print(sum(dict_all_result.values()))

# plot
# plot_pie_all_num_types_cs(dict_all_result)
plt.figure(figsize=(8, 6))

species = (
    "0 $Cs^{+}$",
    "1 $Cs^{+}$",
    "2 $Cs^{+}$",
    "3 $Cs^{+}$"
)

# wet
weight_counts = {
    "Below": np.array([dict_all_result['num_cs_0'],
                       dict_all_result['num_cs_1_d8r'],
                       dict_all_result['num_cs_1_d8r_1_s6r'],
                       dict_all_result['num_cs_1_d8r_2_s6r']]),
    "Above": np.array([0,
                       dict_all_result['num_cs_1_s6r'],
                       dict_all_result['num_cs_2_s6r'],
                       dict_all_result['num_cs_2_d8r_1_s6r']]),
}

# # dry
# weight_counts = {
#     "Below": np.array([dict_all_result['num_cs_0'],
#                        dict_all_result['num_cs_1_d8r'],
#                        dict_all_result['num_cs_1_d8r_1_s6r'],
#                        dict_all_result['num_cs_1_d8r_2_s6r']]),
#     "Above": np.array([0,
#                        dict_all_result['num_cs_1_s6r'],
#                        dict_all_result['num_cs_2_s6r'],
#                        dict_all_result['num_cs_2_d8r_1_s6r']]),
# }

width = 0.7

fig, ax = plt.subplots()
bottom = np.zeros(4)

for boolean, weight_count in weight_counts.items():
    p = ax.bar(species, weight_count, width, label=boolean, bottom=bottom)
    bottom += weight_count

# # add text
# plt.text(0, 50, 'None', horizontalalignment='center', fontsize=9)
# plt.text(1, 10, 'S8R', horizontalalignment='center', fontsize=9)
# plt.text(2, 10, 'S8R & S6R', horizontalalignment='center', fontsize=9)
# plt.text(3, 5, 'S8R & 2 S6R', horizontalalignment='center', fontsize=9)
# plt.text(1, 100, 'S6R', horizontalalignment='center', fontsize=9)
# plt.text(2, 75, '2 S6R', horizontalalignment='center', fontsize=9)


# ax.set_title("Number and types of Cs cations interacting with CO2")
# ax.legend(loc="upper right")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(ymax=700)
plt.xlabel('Numbers of interacting $Cs^{+}$', size=12)
plt.ylabel('Numbers of $CO_{2}-Cs^{+}$', size=12)
plt.tight_layout()
plt.title('Cs-RHO(I43m), '+ '$2CO_{2}$', size=14)
plt.tight_layout()
# plt.savefig(result_folder/'stackbar_2co2_cs_dry(i43m).pdf')
plt.show()


