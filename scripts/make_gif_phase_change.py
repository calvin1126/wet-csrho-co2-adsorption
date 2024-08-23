import sys, os
from pathlib import Path
import glob
from PIL import Image
from ase.visualize.plot import plot_atoms
from spglib import standardize_cell
import matplotlib.pyplot as plt
from md_utils import *
from md_tools import *
from md_helper import get_indices, dict_to_atoms
from ase import Atoms
from ase.visualize import view

def make_gif(frame_folder):
    frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/*.png")]
    print(frames)
    frame_one = frames[0]
    frame_one.save("my_md.gif", format="GIF", append_images=frames, save_all=True, duration=100, loop=0)

all_filepath = ['./data/phase_change/vasprun_6co2.xml']
source_folder = Path('./src')
file_cation = source_folder/'cation_positions_wet.cif'
result_folder = Path('./results/fullopt_6co2')

# #--show the zeolite framework change using gif
# for i, filepath in enumerate(all_filepath):
#     traj = io.read(filepath, ':')
#     atoms_cation_positions = io.read(file_cation)
#     atoms = traj[-1]

#     cnt = 0
#     for atoms in traj:
#         atoms, indices = update_indices(atoms, atoms_cation_positions)
#         # print(indices)
#         atoms_tsite = atoms[indices['Si']] + atoms[indices['Al']]
#         atoms_O = atoms[indices['O'][0:96]]
#         # atoms_water = atoms[indices['O_{H2O}']] + atoms[indices['H_{H2O}']]
#         # atoms_CO2 = atoms[indices['O_{CO2}']] + atoms[indices['C_{CO2}']]
#         atoms = atoms_tsite + atoms_O

#         mystery_cell = (atoms.cell, atoms.get_scaled_positions(), atoms.numbers)
#         new_cell = standardize_cell(mystery_cell, to_primitive=False, symprec=5e-3)

#         new_unit_cell, new_scaled_positions, new_numbers = new_cell
#         new_atoms = Atoms(new_numbers, cell=new_unit_cell, scaled_positions=new_scaled_positions)
#         new_atoms = new_atoms * [2,1,1]

#         fig, ax = plt.subplots()
#         plot_atoms(new_atoms)  # , rotation='90x,90y')

#         # plt.savefig(f"{result_folder}/{cnt}.png")
#         cnt += 1
#         # plt.show()

make_gif(f"{result_folder}")


