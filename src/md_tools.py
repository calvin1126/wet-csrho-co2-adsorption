# date: Oct 24, 2023
# @author:Kun-Lin Wu

import sys, os
sys.path.insert(0,'/Users/calvin11/Library/CloudStorage/Box-Box/project_CO2/src')
from md_utils import *
from ase import io
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from md_helper import get_indices, dict_to_atoms
from ase import neighborlist


def decide_cases(filepath):
    '''
    decide the case by the NAME of the filepath. my_color output is mainly for plotting purpose.
    :param filepath: state the filepath at the main section. Note that the NAME of filepath would affect this function.
    :return: case, my_color
    '''

    if 'no' in filepath:
        case = 'w/o_CO2'
        my_color = 'blue'
    else:
        case = 'w/_CO2'
        my_color = 'orange'

    return case, my_color


def update_cs_indices(atoms, indices):
    '''
    specifically update the Cs atoms in indices dictionary to identify the cs atoms that are located at d8r or s8r sites.
    :param atoms: define at the main code
    :param indices: from update_indices function
    :return: indices
    '''

    # identify Cs atoms that are near the d8r sites and far
    if len(indices['Cs']) != 6:
        indices_cs1 = []
        indices_cs2 = []
        for i_cs in indices['Cs']:
            d1 = np.min(atoms.get_distances(i_cs, indices['S'], mic=True))
            d2 = np.min(atoms.get_distances(i_cs, indices['F'], mic=True))
            # print(d1)
            # print(d2)
            if d1 < d2:
                indices_cs1.append(i_cs)
            else:
                indices_cs2.append(i_cs)
        # print(indices_cs1)
        # print(indices_cs2)

        indices.update({'Cs_{d8r}': indices_cs1})
        indices.update({'Cs_{s6r}': indices_cs2})

    # del indices['Cs']

    return indices


def update_indices(atoms, atoms_cation_positions):
    '''
    update the indices is with cation positions input from atoms_cation_positions cif file.
    :param atoms: define at the main code
    :param atoms_cation_positions: a cif file with the atoms at targeted locations
    :return: atoms, indices
    '''

    new_atoms = atoms + atoms_cation_positions
    del atoms
    atoms = new_atoms

    list_elements = np.unique(atoms.get_chemical_symbols())
    indices = get_indices(atoms)
    for ele in list_elements:
        indices_temp = [a.index for a in atoms if a.symbol == ele]
        indices.update({ele: indices_temp})
        del indices_temp

    indices = update_cs_indices(atoms, indices)

    if len(indices['Cs_{d8r}']) != 6:
        # wet
        indices.update({'Cs_{d8r}': [159, 160, 161, 162, 163, 164]})
        indices.update({'Cs_{s6r}': [165, 166, 167, 168]})
        # dry
        # indices.update({'Cs_{d8r}': [148, 149, 150, 151, 152, 153]})
        # indices.update({'Cs_{s6r}': [154, 155, 156, 157]})

    # this is to differentiate the 2o atoms on co2
    threshold_CO2_distance = 2.0  # Ang
    co2_arr = np.array(indices['O_{CO2}'])

    index_o1_co2 = []  # the o atom on co2 that is closer to the cs atom
    index_o2_co2 = []  # the other o atom on co2
    for i in indices['C_{CO2}']:
        d = atoms.get_distances(i, indices['O_{CO2}'], mic=True)
        indices_O_CO2 = d < threshold_CO2_distance
        # indices_to_use = np.where(indices_O_CO2)[0]
        # print(i, co2_arr[indices_O_CO2])
        index_o1_co2.append(co2_arr[indices_O_CO2][0])  # identify by its naming order
        index_o2_co2.append(co2_arr[indices_O_CO2][1])
    indices.update({'OCO2_{nearCs}': index_o1_co2})  # update the indices dictionary
    indices.update({'OCO2_{farCs}': index_o2_co2})

    # print(indices)
    # sys.exit()

    # update dict with the name of positions instead of using substitute atoms
    indices['d8r'] = indices.pop('S')
    indices['s8r'] = indices.pop('Cl')
    indices['s6r'] = indices.pop('F')
    indices['s4r'] = indices.pop('He')

    # assert len(indices['Cs_{d8r}']) == 6
    assert len(indices['d8r']) == 6
    assert len(indices['P']) == 2
    # assert len(indices['H']) == len(indices['O_{H2O}']) * 2

    return atoms, indices


def find_avg_num_close_atoms(traj, ele1, ele2):
    '''
    find the avg number of ele2 atoms that are closed to the ele1 atoms by natural cutoff built by ase
    :param traj: define at the main section
    :param ele1: define at the main section
    :param ele2: define at the main section
    :return: avg_num
    '''

    list_num_all = []
    for atoms in traj:
        # get the number of water bound to Cs atoms
        atoms, indices = update_indices(atoms, atoms_cation_positions)
        # print(indices)
        cutOff = neighborlist.natural_cutoffs(atoms)
        nl = NeighborList(cutOff, bothways=True, self_interaction=False)
        nl.update(atoms)
        list_num = []
        for i in indices[ele1]:
            indices_near, offsets = nl.get_neighbors(i)
            indices_nearby = []
            for j in indices_near:
                for k in indices[ele2]:
                    if k == j:
                        indices_nearby.append(k)
            # print(indices_nearby)
            list_num.append(len(indices_nearby))
        # print(list_num)

        #     print('Cs atom #', i, 'bound to', len(indices_nearby), 'of water molecules')
        list_num_all.append(list_num)
    # print(list_num_all)
    avg_num = np.mean(list_num_all, axis=0, dtype=object)
    # print(avg_num)

    return avg_num


def analyze_single_snapshot(atoms, indices, ele1, ele2):
    '''

    :param atoms:
    :param indices:
    :param ele1:
    :param ele2:
    :return:
    '''

    list_d = []
    list_d_min = []
    for i in indices[ele1]:
        d = atoms.get_distances(i, indices[ele2], mic=True)
        d_min = np.min(d)
        list_d.append(d)
        list_d_min.append(d_min)
        # print(i,d_min,d)
    d_all = np.concatenate(list_d)
    d_min_all = list_d_min

    return d_all, d_min_all


def analyze_all_snapshots(traj, ele1, ele2):
    '''

    :param traj:
    :param ele1:
    :param ele2:
    :return:
    '''

    list_d = []
    list_d_min = []
    traj_to_use = traj
    num_snapshots = len(traj_to_use)
    for atoms in traj_to_use:
        atoms, indices = update_indices(atoms, atoms_cation_positions)
        d_all, d_min_all = analyze_single_snapshot(atoms, indices, ele1, ele2)
        list_d.append(d_all)
        list_d_min.append(d_min_all)
        num_ele1 = len(indices[ele1])
        num_ele2 = len(indices[ele2])
        assert len(d_all) == num_ele1 * num_ele2
        assert len(d_min_all) == num_ele1

    d = np.concatenate(list_d)
    d_min = np.concatenate(list_d_min)
    assert len(d) == num_ele1 * num_ele2 * num_snapshots
    assert len(d_min) == num_ele1 * num_snapshots

    return d, d_min


def plot_dmin_data(d_min, ele1, ele2, case, my_color):
    '''

    :param d_min:
    :param ele1:
    :param ele2:
    :param case:
    :param my_color:
    :return:
    '''

    # --plot min data
    plt.figure(figsize=(6, 3))
    # plt.subplot(2,1,2)
    ydata = d_min
    plt.hist(ydata, bins=50, density=1, alpha=0.5, label=case, color=my_color)
    plt.title('$%s-%s$, %s total distances, %s total snapshots' % (ele1, ele2, len(ydata), len(traj)))
    plt.xlabel('$d(%s-%s)$, ${\AA}$' % (ele1, ele2))
    plt.ylabel('Frequency')
    plt.xlim(xmin=0.00, xmax=8.0)
    #plt.yticks([0, 1, 2])
    plt.legend()
    plt.tight_layout()
    plt.show()
    # plt.savefig('/global/homes/k/klw1126/UCB/fig_wet_%s_%s_distance_210730.pdf' % (ele1,ele2),dpi=300)


def plot_2d_histogram(d_min_dict, case):
    '''

    :param d_min_dict:
    :param case:
    :return:
    '''

    plt.figure(figsize=(6, 5))
    xdata = d_min_dict[('Cs_{d8r}', 'O_H2O')]
    ydata = d_min_dict[('Cs_{d8r}', 'd8r')]

    x_min = np.min(xdata)
    x_max = np.max(xdata)
    y_min = np.min(ydata)
    y_max = np.max(ydata)

    x_bins = np.linspace(x_min, x_max, 100)
    y_bins = np.linspace(y_min, y_max, 100)

    plt.hist2d(xdata, ydata, bins=[x_bins, y_bins], density=True, label=case)

    plt.xlabel('$d(Cs-O_H2O)$, ${\AA}$')
    plt.ylabel('$d(Cs-d8R_{center})$, ${\AA}$')

    plt.colorbar()
    plt.show()