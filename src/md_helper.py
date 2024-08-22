# date: Oct 24, 2023
# @author:Kun-Lin Wu

from ase import io
import numpy as np
from ase.visualize import view
from ase.spacegroup import crystal

def get_indices(atoms):
    indices = {}    # create a dictionary of indices for targeted atoms

    # get the indices by types of atoms
    indices_Si = [a.index for a in atoms if a.symbol == 'Si']
    indices_Al = [a.index for a in atoms if a.symbol == 'Al']
    indices_H = [a.index for a in atoms if a.symbol == 'H']
    indices_O = [a.index for a in atoms if a.symbol == 'O']
    indices_C = [a.index for a in atoms if a.symbol == 'C']
    indices_Cs = [a.index for a in atoms if a.symbol == 'Cs']

    threshold_CO2_distance = 1.30
    threshold_H2O_distance = 1.20

    # identify the O atoms on H2O
    indices_O_H2O = []
    for i in indices_H:
        d = atoms.get_distances(i, indices_O, mic=True)
        for count, value in enumerate(d):
            if value < threshold_H2O_distance:  # hardcoding
                indices_O_H2O.append(count)
                # print([count,value])
                # print(value)

    indices_O_H2O = list(set(indices_O_H2O))
    indices_OH2O = []
    for i in indices_O_H2O:
        indices_OH2O.append(indices_O[i])
    # print(indices_O_H2O)
    indices.update({'O_H2O': indices_OH2O})
    # assert len(indices_H) == 2 * len(indices_OH2O)

    # identify the O atoms on CO2
    indices_O_CO2 = []
    for i in indices_C:
        d = atoms.get_distances(i, indices_O, mic=True)
        for count, value in enumerate(d):
            if value < threshold_CO2_distance:  # hardcoding
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


    indices = {
    'O_{H2O}': indices_OH2O,
    'H_{H2O}': indices_H,
    'O_{CO2}': indices_OCO2,
    'C_{CO2}': indices_C,
    'Cs': indices_Cs,
    'Si': indices_Si,
    'Al': indices_Al}

    return indices


def dict_to_atoms(dict, elements, atoms, traj, types_to_include):
    # this is to create atoms from a dictionary of atoms with their positions
    # types_to_include = [] or 'all'
    if 'all' in types_to_include:       # include all types of atoms
        for key, val in dict.items():
            ele = elements[key]
            atoms = crystal(ele, dict[key], spacegroup=229, cell = atoms.cell)
            traj.append(atoms)
    else:                               # include only the types of selected atoms
        for key, val in dict.items():
            if key in types_to_include:
                ele = elements[key]
                atoms = crystal(ele, dict[key], spacegroup=229, cell=atoms.cell)
                traj.append(atoms)

    final_atoms = traj[0]
    for atoms in traj[1:]:
        final_atoms  = final_atoms + atoms

    final_atoms = final_atoms + traj[-1]

    return (final_atoms)



