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


def identify_cations_interacting_co2(i_ele, ele2, atoms):
    '''
    Identify which Cs cations are interacting with CO2 based on the O-Cs distance. In this case, the threshold for the
    distance is 3.7 angstrom.

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
    atom_d8r = atoms[[indices['d8r'][dmin_index_d8r]]]
    # view(atom_d8r)
    position_d8r = atom_d8r.get_positions()


    # visualize the system
    atoms_tsite = atoms[indices['Si']] + atoms[indices['Al']]
    atoms_water = atoms[indices['O_{H2O}']] + atoms[indices['H_{H2O}']]
    atoms_OCO2 = atoms[[indices['OCO2_{nearCs}'][i_ele]]] + atoms[[indices['OCO2_{farCs}'][i_ele]]]  # !!!need to renaming the OCO2

    if (bool(o_cs_index) == True):
        target_cs_indices = [val for key, val in enumerate(indices[ele2]) if key in set(o_cs_index)]
        # print(indices)
        # print(target_cs_indices)
        # sys.exit()
        atoms_target_cs = atoms[target_cs_indices]
        atoms_target = atoms[[indices[ele1][i_ele]]] + atoms_target_cs + atoms_tsite + atoms_OCO2 + atoms[[indices['Cs_{d8r}'][5]]] + atoms_water # iele
        pos_0 = position_d8r
        center_of_cell = np.dot([0.5, 0.5, 0.5], atoms_target.cell)
        vec = center_of_cell - pos_0
        atoms_target.translate(vec)
        atoms_target.wrap()

    else:
        atoms_target = atoms[[indices[ele1][i_ele]]] + atoms_tsite + atoms_OCO2 + atoms[[indices['Cs_{d8r}'][5]]] + atoms_water # iele
        pos_0 = position_d8r
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




# Settings for wet and dry cases
case = 'wet'  # Change to 'wet' for wet case

if case == 'wet':
    data_folder = Path('./data/csrho-im3m')
    filepath = data_folder / 'long_md_6co2_15h2o.traj'
    source_folder = Path('./src')
    file_cation = source_folder / 'cation_positions_wet.cif'
    
elif case == 'dry':
    data_folder = Path('./data/csrho-i43m')
    filepath = data_folder / 'long_md_2co2.traj'
    source_folder = Path('./src')
    file_cation = source_folder / 'cation_positions_dry.cif'


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
# for i_ele in range(len(indices[ele1])):
traj_target_wrapped, num_cs_total, num_cs_d8r, num_cs_s6r = classify_interacting_cs(traj, 0, ele2)  #0 for SI1, 5 for SI2
view(traj_target_wrapped)
