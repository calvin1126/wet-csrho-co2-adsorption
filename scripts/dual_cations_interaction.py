# date: Oct 25, 2023
# @author:Kun-Lin Wu

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

# !!!! check with dry data

def identify_cations_interacting_co2(i_ele, ele2, atoms):
    '''
    Identify which Cs cations are interacting with CO2 based on the O-Cs distance. In this case, the threshold for the
    distance is 4.0 angstrom.

    Parameters:
    i_ele: the indices of ele1.
    ele2: the cations that are targeted.
    '''

    # check the first O atom in CO2 that is within the interacting distance range with Cs
    dist_o1_cs = atoms.get_distances(indices['OCO2_{nearCs}'][i_ele], indices[ele2], mic=True)         #!!!need to renaming the OCO2
    dist_select_o1 = [i_dist for i_dist in dist_o1_cs if i_dist < 3.7]
    dist_select_index_o1 = [key for key, val in enumerate(dist_o1_cs) if val in set(dist_select_o1)]
    # print('o1_cs:', dist_select_o1)
    # print('o1_cs_index:', dist_select_index_o1)

    # check the second O atom in CO2 that is within the interacting distance range with Cs
    dist_o2_cs = atoms.get_distances(indices['OCO2_{farCs}'][i_ele], indices[ele2], mic=True)          #!!!need to renaming the OCO2
    dist_select_o2 = [i_dist for i_dist in dist_o2_cs if i_dist < 3.7]
    dist_select_index_o2 = [key for key, val in enumerate(dist_o2_cs) if val in set(dist_select_o2)]
    # print('o2_cs:', dist_select_o2)
    # print('o2_cs_index:', dist_select_index_o2)

    o_cs_index_combine = dist_select_index_o1 + dist_select_index_o2
    o_cs_index = [*set(o_cs_index_combine)]  # this is to prevent duplicate numbers
    # print('The index of Cs that interacts with CO2: ', o_cs_index)

    return o_cs_index


def get_all_targeted_atoms(i_ele, ele1, ele2, atoms):
    '''
    Get the targeted atoms of the framework, Cs, co2, h2o, corresponding d8r site. Initially, find the closest d8r of
    co2 and transform the cell based on it as the center. Including all the cs+ cations, water and co2 based
    on indices.

    Parameters:
    i_ele: the indices of O_CO2.
    ele2: the cations that are targeted.
    '''

    # determine the corresponding d8r site of co2 initially
    d_csd8r_d8r = atoms.get_distances(indices['Cs_{d8r}'][i_ele], indices['d8r'], mic=True)  # iele
    dmin_csd8r_d8r = np.min(d_csd8r_d8r)
    dmin_index_d8r = np.where(d_csd8r_d8r == dmin_csd8r_d8r)[0][0]

    # visualize the system
    atoms_tsite = atoms[indices['Si']] + atoms[indices['Al']]
    atoms_OCO2 = atoms[[indices['OCO2_{nearCs}'][i_ele]]] + atoms[[indices['OCO2_{farCs}'][i_ele]]]    #!!!need to renaming the OCO2
    atoms_target = atoms[[indices['d8r'][dmin_index_d8r]]] + atoms[[indices[ele1][i_ele]]] + atoms[indices[ele2]] + \
                   atoms_tsite + atoms_OCO2 + atoms[indices['O_{H2O}']] + atoms[indices['H_{H2O}']]    #!!!need to renaming the OCO2
    pos_0 = atoms_target.get_positions()[0]
    center_of_cell = np.dot([0.5, 0.5, 0.5], atoms_target.cell)
    vec = center_of_cell - pos_0
    atoms_target.translate(vec)
    atoms_target.wrap()

    return atoms_target


def get_targeted_cs_atoms(i_ele, ele1, ele2, o_cs_index, atoms):
    '''
    Get the targeted atoms of the framework, co2, h2o, corresponding d8r site, and the interacting Cs cations.
    Initially, find the closest d8r of co2 and transform the cell based on it as the center. Include the interacting
    cs+ cations, water, and co2 based on indices.

    Parameters:
    i_ele: the indices of O_CO2.
    ele2: the cations that are targeted.
    '''

    # determine the corresponding d8r site of co2 initially
    d_csd8r_d8r = atoms.get_distances(indices['Cs_{d8r}'][i_ele], indices['d8r'], mic=True)  # iele
    dmin_csd8r_d8r = np.min(d_csd8r_d8r)
    dmin_index_d8r = np.where(d_csd8r_d8r == dmin_csd8r_d8r)[0][0]

    # visualize the system
    atoms_tsite = atoms[indices['Si']] + atoms[indices['Al']]
    atoms_OCO2 = atoms[[indices['OCO2_{nearCs}'][i_ele]]] + atoms[[indices['OCO2_{farCs}'][i_ele]]]  # !!!need to renaming the OCO2

    if (bool(o_cs_index) == True):
        target_cs_indices = [val for key, val in enumerate(indices[ele2]) if key in set(o_cs_index)]
        atoms_target_cs = atoms[target_cs_indices]
        atoms_target = atoms[[indices['d8r'][dmin_index_d8r]]] + atoms[[indices[ele1][i_ele]]] + atoms_target_cs \
                       + atoms_tsite + atoms_OCO2  # iele
        pos_0 = atoms_target.get_positions()[0]
        center_of_cell = np.dot([0.5, 0.5, 0.5], atoms_target.cell)
        vec = center_of_cell - pos_0
        atoms_target.translate(vec)
        atoms_target.wrap()

    else:
        atoms_target = atoms[[indices['d8r'][dmin_index_d8r]]] + atoms[[indices[ele1][i_ele]]] + atoms_tsite + atoms_OCO2  # iele
        pos_0 = atoms_target.get_positions()[0]
        center_of_cell = np.dot([0.5, 0.5, 0.5], atoms_target.cell)
        vec = pos_0 - center_of_cell
        atoms_target.translate(-1 * vec)
        atoms_target.wrap()

    return atoms_target


def classify_interacting_cs(traj, i_ele, ele2, all=False):
    '''
    Classify the types of Cs cations that CO2 is interacting with.

    Parameters:
        traj: the traj file that are being analyzed
        i_ele: the element of CO2
        ele2: the cations
        all=True if only showing the interacting Cs+ instead of all the Cs

    Return:
        traj_target_wrapped: visualize the targeted atoms
        num_cs_total: total number of cs cations interacting with CO2 over traj
        num_cs_d8r: number of cs@d8r cations interacting with CO2 over traj
        num_cs_s6r: number of cs@s6r cations interacting with CO2 over traj
    '''

    traj_target_wrapped = []
    num_cs_total = []
    num_cs_d8r = []
    num_cs_s6r = []
    for atoms in traj:
        atoms, indices = update_indices(atoms, atoms_cation_positions)
        o_cs_index = identify_cations_interacting_co2(i_ele, ele2, atoms)  # identify cs+ that are interacting with CO2

        if all == True:
            # get the targeted atoms with all the Cs cations
            atoms_target = get_all_targeted_atoms(i_ele, ele1, ele2, atoms)
            traj_target_wrapped.append(atoms_target)
        else:
            # get the targeted atoms with only the interacting Cs cations
            atoms_target = get_targeted_cs_atoms(i_ele, ele1, ele2, o_cs_index, atoms)
            traj_target_wrapped.append(atoms_target)

        # classify the cs that co2 is interacting with
        target_cs_indices = [val for key, val in enumerate(indices[ele2]) if key in set(o_cs_index)]
        num_cs_total.append(len(target_cs_indices))
        cs_d8r_co2 = [val for key, val in enumerate(indices['Cs_{d8r}']) if val in set(target_cs_indices)]
        cs_s6r_co2 = [val for key, val in enumerate(indices['Cs_{s6r}']) if val in set(target_cs_indices)]
        # print('The number of Cs atoms that binds with CO2', len(target_cs_indices))
        # print('The Cs near d8r center that binds with CO2:', cs_d8r_co2)
        # print('The number of Cs-d8r that binds with CO2:', len(cs_d8r_co2))
        num_cs_d8r.append(len(cs_d8r_co2))
        # print('The Cs near s6r center that binds with CO2:', cs_s6r_co2)
        # print('The number of Cs-s6r that binds with CO2:', len(cs_s6r_co2))
        num_cs_s6r.append(len(cs_s6r_co2))

    return traj_target_wrapped, num_cs_total, num_cs_d8r, num_cs_s6r


def classify_num_cs_interacting(num_cs_total):
    '''
    Classify the number of cs interacting based on the list, num_cs_total.

    Parameters:
        num_cs_total: total number of cs cations interacting with CO2 over traj
    '''

    # num_cs_total.sort()  # order the list from the smallest to the largest
    # print('Total number of Cs interacting with CO2: ', num_cs_total)
    dict_num_cs = {}
    for i_num_cs in range(4):
        dict_num_cs['num_cs_%s' %(i_num_cs)] = num_cs_total.count(i_num_cs)
    # print(dict_num_cs)

    return dict_num_cs


def classify_type_cs_interacting(num_cs_total, num_cs_d8r, num_cs_s6r):
    '''
    Classify the types of cs, cs@d8r and cs@s6r interacting based on the list, num_cs_total, num_cs_d8r and num_cs_s6r.

    Parameters:
        num_cs_total: total number of cs cations interacting with CO2 over traj
        num_cs_d8r: number of cs@d8r cations interacting with CO2 over traj
        num_cs_s6r: number of cs@s6r cations interacting with CO2 over traj
    '''

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

    # print(dict_types_cs)

    return dict_types_cs


def plot_pie_num_types_cs(i_ele, dict_num_cs, dict_types_cs):
    '''
    Plot pie chart of the num and types of Cs cations that are interacting with CO2

    Parameters:
        dict_num_cs: dictionary of the number of interacting Cs cations
        dict_types_cs: dictionary of the types of interacting Cs cations
    '''

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    labels = ['0 Cs', '1 Cs', '2 Cs', '3 Cs']
    sizes = [dict_num_cs['num_cs_0'],
             dict_num_cs['num_cs_1'],
             dict_num_cs['num_cs_2'],
             dict_num_cs['num_cs_3']]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.title('The number of Cs cations interacting with the CO2[%s]' % (i_ele))  # iele

    if (dict_num_cs['num_cs_1'] != 0):
        plt.subplot(2, 2, 2)
        labels = ['S8R', 'S6R']
        sizes = [dict_types_cs['num_cs_1_d8r'],
                 dict_types_cs['num_cs_1_s6r']]
        plt.pie(sizes, labels=labels, autopct='%1.1f%%')
        plt.title('Types of Cs cations when there is 1 Cs')

    if (dict_num_cs['num_cs_2'] != 0):
        plt.subplot(2, 2, 3)
        labels = ['2 S8R', '2 S6R', '1 S8R+ 1 S6R']
        sizes = [dict_types_cs['num_cs_2_d8r'],
                 dict_types_cs['num_cs_2_s6r'],
                 dict_types_cs['num_cs_1_d8r_1_s6r']]
        plt.pie(sizes, labels=labels, autopct='%1.1f%%')
        plt.title('Types of Cs cations when there are 2 Cs')

    if (dict_num_cs['num_cs_3'] != 0):
        plt.subplot(2, 2, 4)
        labels = ['3 S8R', '3 S6R', '1 S8R+ 2 S6R', '2 S8R+ 1 S6R']
        sizes = [dict_types_cs['num_cs_3_d8r'],
                 dict_types_cs['num_cs_3_s6r'],
                 dict_types_cs['num_cs_1_d8r_2_s6r'],
                 dict_types_cs['num_cs_2_d8r_1_s6r']]
        plt.pie(sizes, labels=labels, autopct='%1.1f%%')
        plt.title('Types of Cs cations when there are 3 Cs')

    # plt.savefig('../results/00_co2_pos02/num_cs_co2_%s_pie.pdf' % (i_ele))
    # plt.savefig('../results/01_co2_250traj/num_cs_co2_%s_pie.pdf' % (i_ele))
    plt.show()


def plot_pie_all_num_types_cs(dict_all_result):
    '''
    plot pie chart of the "all" cs num interacting with co2

    Parameters:
        dict_all_result: dictionary containing the total numbers and types of cations interacting with CO2
    '''

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    labels = ['0 Cs', '1 Cs', '2 Cs', '3 Cs']
    sizes = [dict_all_result['num_cs_0'],
             dict_all_result['num_cs_1'],
             dict_all_result['num_cs_2'],
             dict_all_result['num_cs_3']]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.title('The total number of Cs cations interacting with the CO2')

    if (dict_all_result['num_cs_1'] != 0):
        plt.subplot(2, 2, 2)
        labels = ['S8R', 'S6R']
        sizes = [dict_all_result['num_cs_1_d8r'],
                 dict_all_result['num_cs_1_s6r']]
        plt.pie(sizes, labels=labels, autopct='%1.1f%%')
        plt.title('Types of Cs cations when there is 1 Cs')

    if (dict_all_result['num_cs_2'] != 0):
        plt.subplot(2, 2, 3)
        labels = ['2 S8R', '2 S6R', '1 S8R+ 1 S6R']
        sizes = [dict_all_result['num_cs_2_d8r'],
                 dict_all_result['num_cs_2_s6r'],
                 dict_all_result['num_cs_1_d8r_1_s6r']]
        plt.pie(sizes, labels=labels, autopct='%1.1f%%')
        plt.title('Types of Cs cations when there are 2 Cs')

    if (dict_all_result['num_cs_3'] != 0):
        plt.subplot(2, 2, 4)
        labels = ['3 S8R', '3 S6R', '1 S8R+ 2 S6R', '2 S8R+ 1 S6R']
        sizes = [dict_all_result['num_cs_3_d8r'],
                 dict_all_result['num_cs_3_s6r'],
                 dict_all_result['num_cs_1_d8r_2_s6r'],
                 dict_all_result['num_cs_2_d8r_1_s6r']]
        plt.pie(sizes, labels=labels, autopct='%1.1f%%')
        plt.title('Types of Cs cations when there are 3 Cs')

    # plt.savefig('../results/00_co2_pos02/num_cs_co2_all_pie.pdf')
    # plt.savefig('../results/01_co2_250traj/num_cs_co2_all_pie.pdf')
    plt.show()



# files

# data file for wet Cs/RHO: The traj file is wet Cs/RHO (im3m, 6CO2, 15H2O).
# The traj file is the result after a combination of 5 ensembles of different Al distributions.
# data_folder = Path('/Users/calvin11/Library/CloudStorage/Box-Box/project_WetCsRHO_CO2_adsorption/data/02_im3m')
# filepath = data_folder/'im3m_cs10_6co2_15h2o_250.traj'

# data file for dry Cs/RHO: The traj file is dry Cs/RHO (i43m, 2CO2).
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
ele1 = 'C_{CO2}'   # edit: the element
ele2 = 'Cs'        # edit: the element


#--main
traj = io.read(filepath, ':')
atoms_cation_positions = io.read(file_cation)
atoms = traj[0]
atoms, indices = update_indices(atoms, atoms_cation_positions)
# view(atoms)
print('Total snapshots analyzed %s' % len(traj))

dict_all_cs = []
for i_ele in range(len(indices[ele1])):
    traj_target_wrapped, num_cs_total, num_cs_d8r, num_cs_s6r = classify_interacting_cs(traj, i_ele, ele2)
    # print('Number of Cs cations interacting with CO2: ', num_cs_total)
    # print('Number of Cs(S8R) interacting with CO2: ', num_cs_d8r)
    # print('Number of Cs(S6R) interacting with CO2: ', num_cs_s6r)
    # view(traj_target_wrapped)

    # get the dictionary of the numbers and types of Cs cations interacting with CO2
    dict_num_cs = classify_num_cs_interacting(num_cs_total)
    dict_types_cs = classify_type_cs_interacting(num_cs_total, num_cs_d8r, num_cs_s6r)

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

# add text
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
plt.ylim(ymax=850)
plt.xlabel('Numbers of interacting $Cs^{+}$', size=12)
plt.ylabel('Numbers of $CO_{2}-Cs^{+}$', size=12)
plt.title('Cs-RHO(Im3m), '+ '$6CO_{2} + 15H_{2}O$', size=14) #$2CO_{2}$
plt.tight_layout()
# plt.savefig(result_folder/'stackbar_co2_cs_wet_thres3.6.png')
plt.show()


