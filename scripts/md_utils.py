# date: Oct 24, 2023
# @author:Kun-Lin Wu


import sys, os
path_home = os.environ['HOME']
sys.path.insert(0, path_home + '/lib/zeolite_tools')
from ase import io
import numpy as np
from glob import glob
from ase.constraints import FixAtoms
from ase.calculators.vasp import Vasp
# from identify_zeolite_atoms import analyze_zeolite_atoms
from ase.visualize import view

from ase.neighborlist import NeighborList

try:
    from ase.utils import natural_cutoffs
except:
    from ase.neighborlist import natural_cutoffs
import itertools
from ase import Atoms
import random


# %%
def get_alternate_si_al_structure(atoms):
    # Making a structure with Si/Al = 1 (if possible)

    atoms_t = atoms[[a.index for a in atoms if a.symbol in ['Al', 'Si']]]
    atoms_o = atoms[[a.index for a in atoms if a.symbol == 'O']]
    atoms_Cs = atoms[[a.index for a in atoms if a.symbol == 'Cs']]
    atoms_H = atoms[[a.index for a in atoms if a.symbol == 'H']]
    atoms_C = atoms[[a.index for a in atoms if a.symbol == 'C']]

    # Because we like working with the variable atoms.
    atoms = atoms_t
    num_atoms = len(atoms)
    nl = NeighborList([3.4 / 2] * num_atoms, bothways=True, self_interaction=False)
    nl.update(atoms)
    atoms.set_tags([0] * num_atoms)

    # Keep the zeorth atom as Si
    current_list = [0]
    atoms[0].tag = 1
    atoms[0].symbol = 'Si'
    old_si_al_ratio = 10000

    # current_list defines the Si atoms that we're working with
    count = 0
    while count < 10:
        # print('Iteration %s:' %count)
        # print('\tstep0, length current list = %s' % len(current_list))
        for index in current_list:
            current_neighs, offsets = nl.get_neighbors(index)
            # current_neighs are assigned as Al
            assert len(current_neighs) == 4
            for i_neigh in current_neighs:
                atoms[i_neigh].symbol = 'Al'
                atoms[i_neigh].tag = 1
        # print('\tstep1, assigned = %s' % sum(atoms.get_tags()))

        new_list = []
        for index in current_list:
            current_neighs, offsets = nl.get_neighbors(index)
            for i_neigh2 in current_neighs:
                neighs, offsets = nl.get_neighbors(i_neigh2)
                for val in neighs:
                    if atoms[val].tag == 0:
                        new_list.append(val)
                        atoms[val].tag = 1
        new_list = np.unique(new_list)
        # print('\tstep2, assigned = %s' % sum(atoms.get_tags()))

        # Finish iteration
        count = count + 1
        # traj.append(atoms)
        # print(atoms)
        current_list = new_list

        count_Al = len([a.index for a in atoms if a.symbol == 'Al'])
        count_Si = len([a.index for a in atoms if a.symbol == 'Si'])

        indices_Al = [a.index for a in atoms if a.symbol == 'Al']
        distance_Al = atoms.get_distance(indices_Al[0], indices_Al[1])

        ratio_si_by_al = count_Si / count_Al
        # print('Ratio Si/Al=',ratio_si_by_al)
        # print('Al-Al distances=', distance_Al)

    # note atoms only has T site for now
    atoms_out = atoms + atoms_o + atoms_Cs + atoms_H + atoms_C  # showing all the atoms in CsRHO

    return atoms_out


# %%
def get_target_si_al_ratio(atoms, target_si_by_al):
    index_t = [a.index for a in atoms if a.symbol in ['Si', 'Al']]
    index_si = [a.index for a in atoms if a.symbol == 'Si']
    index_al = [a.index for a in atoms if a.symbol == 'Al']
    count_t = len(index_t)
    count_si = len(index_si)
    count_al = len(index_al)
    current_si_by_al = count_si / count_al
    # print('Current Si/Al = ', current_si_by_al)
    # print('Target Si/Al = ', target_si_by_al)

    atoms_t = atoms[[a.index for a in atoms if a.symbol == 'Si']]
    count_t = len(index_t)

    # final counts
    count_al_final = int(np.ceil(count_t / (1 + target_si_by_al)))
    count_si_final = int(count_t - count_al_final)
    final_si_by_al = count_si_final / count_al_final
    # print('Final Si/Al = ', final_si_by_al)

    num_al_to_replace = count_al - count_al_final
    indices_al_to_replace = random.sample(index_al, num_al_to_replace)
    for i in indices_al_to_replace:
        assert atoms[i].symbol == 'Al'
        atoms[i].symbol = 'Si'

    test_count_si = len([a.index for a in atoms if a.symbol == 'Si'])
    test_count_al = len([a.index for a in atoms if a.symbol == 'Al'])
    assert test_count_si == count_si_final
    assert test_count_al == count_al_final

    return atoms


def check_al_o_al(atoms):
    '''
    this function is for checking if Al-O-Al exists
    '''
    index_al = [a.index for a in atoms if a.symbol == 'Al']
    index_o = [a.index for a in atoms if a.symbol == 'O']
    # print(index_al)

    # --finding the index of all O atoms that are adjacent to Al atoms
    for i in index_al:
        d = atoms.get_distances(i, index_o)
        index_o_al = []
        for count, value in enumerate(d):
            if value < 1.67:  # hardcoding distance
                index_o_al.append(index_o[count])
        # print(index_o_al)

        # --check the distance of Al with each selected O atoms, if there's
        # only one distance that is less than 1.67, then then it means that
        # no Al-O-Al presence
        for i in index_o_al:
            d_o_al = atoms.get_distances(i, index_al)
            num_o_al = [i for i in d_o_al if i < 1.67]  # hardcoding distance
            assert len(num_o_al) == 1

    return atoms


def identify_O_CO2_H2O(atoms):
    '''
    identify the O atoms in CO2 and H2o molecules
    '''
    indices = {}
    # --indentify H2O
    indices_H = [a.index for a in atoms if a.symbol == 'H']
    indices_O = [a.index for a in atoms if a.symbol == 'O']
    indices_O_H2O = []
    for i in indices_H:
        d = atoms.get_distances(i, indices_O, mic=True)
        for count, value in enumerate(d):
            if value < 1.00:  # hardcoding
                indices_O_H2O.append(count)
                # print([count,value])
                # print(value)

    indices_O_H2O = list(set(indices_O_H2O))
    indices_OH2O = []
    for i in indices_O_H2O:
        indices_OH2O.append(indices_O[i])
    # print(indices_O_H2O)
    indices.update({'O_H2O': indices_OH2O})
    assert len(indices_H) == 2 * len(indices_OH2O)

    # --indentify CO2
    indices_C = [a.index for a in atoms if a.symbol == 'C']
    indices_O = [a.index for a in atoms if a.symbol == 'O']
    indices_O_CO2 = []
    for i in indices_C:
        d = atoms.get_distances(i, indices_O, mic=True)
        for count, value in enumerate(d):
            if value < 1.20:  # hardcoding
                indices_O_CO2.append(count)
                # print([count,value])
                # print(value)

    indices_O_CO2 = list(set(indices_O_CO2))
    indices_OCO2 = []
    for i in indices_O_CO2:
        indices_OCO2.append(indices_O[i])
    # print(indices_O_CO2)
    indices.update({'O_CO2': indices_OCO2})
    assert len(indices_OCO2) == 2 * len(indices_C)

    # print(indices)

    return indices



