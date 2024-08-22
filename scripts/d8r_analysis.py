# date: Nov 11, 2023
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
from ase.spacegroup import get_spacegroup


# files
data_folder = Path('/Users/calvin11/Library/CloudStorage/Box-Box/project_CO2/data/dft_refinement/00_cs/07_d8r_cs')
all_filepath = [
                data_folder/'00_0h2o/opt_400/vasprun.xml',
                data_folder/'01_1h2o/opt_400/vasprun.xml',
                data_folder/'02_2h2o/opt_400/vasprun.xml',
                ]

# save result
result_folder = Path('/Users/calvin11/Library/CloudStorage/Box-Box/project_CO2/results/paper_cs-rho')

cases = ['no water', '1 h2o']
for i_file, filepath in enumerate(all_filepath):
    traj = io.read(filepath, ':')
    atoms = traj[-1]
    # view(atoms)
    # sys.exit()

    # get all the atoms indices
    indices_all_atoms = [a.index for a in atoms]

    # get Cs+ indices in atoms
    indices_cs = [a.index for a in atoms if a.symbol == 'Cs']

    # get Cs+ force in xyz values
    forces_cs = np.concatenate(atoms.get_forces()[indices_cs])
    print('The forces on Cs+ in d8r ring with %s case is:' % (cases[i_file]), forces_cs)

    # We want to look at how the ring shape changes (im3m, centrosymmetric; to i43m, non-centrosymmetric). We pair
    # the O atoms on one of the rings. O atoms with (index 33, 36) are in pair and (index 34, 35) are in another pair.
    d1_list = []
    d2_list = []
    for atoms in traj:
        d1 = atoms.get_distances(indices_all_atoms[33], indices_all_atoms[36],  mic=True)
        d2 = atoms.get_distances(indices_all_atoms[34], indices_all_atoms[35], mic=True)
        d1_list.append(d1[0])
        d2_list.append(d2[0])

    # plot
    plt.figure(figsize=(8, 6))
    xdata = np.linspace(0, len(d1_list), len(d1_list))
    plt.plot(xdata, d1_list, label='d1')
    plt.plot(xdata, d2_list, label='d2')
    plt.xlabel('# traj')
    plt.ylabel('distance, ${\AA}$')
    plt.title('O atoms distance in a ring change over traj')
    plt.tight_layout()
    plt.legend()
    plt.show()


    sys.exit()







