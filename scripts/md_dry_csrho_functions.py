# Import required libraries
import sys
import os
from md_utils import *
from md_tools import *
from ase import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ase.spacegroup import crystal
from md_helper import get_indices, dict_to_atoms
from ase import neighborlist, Atoms
from scipy.stats import norm, alpha
from ase.visualize import view
from ase.build import molecule
from scipy.optimize import curve_fit

# Function Definitions

def get_cs_d8r_dist_dry_vector(atoms, indices):
    """
    Calculate distance vectors between Cs atoms at d8r positions and corresponding sites in dry conditions.

    Parameters:
    - atoms (ase.Atoms): The atomic configuration.
    - indices (dict): A dictionary containing indices of Cs and d8r positions.

    Returns:
    - dict: A dictionary containing distance vectors for each Cs-d8r pair.
    """
    d = {}
    for i in range(6):  # Iterate over the 6 Cs-d8r pairs
        d[i] = atoms.get_distances(indices['d8r'][i], indices['Cs_{d8r}'][i], mic=True, vector=True)
    return d

def get_dict_distance_vector(ele, indices):
    """
    Initialize a dictionary to store distance vectors.

    Parameters:
    - ele (str): The element type (e.g., 'Cs_{d8r}').
    - indices (dict): A dictionary containing indices for the elements.

    Returns:
    - dict: A dictionary initialized to store x, y, and z components of distance vectors.
    """
    dict_distance_vector = {}
    for i_keys in range(len(indices[ele])):
        dict_distance_vector[f'd_vec_x_{i_keys}'] = []
        dict_distance_vector[f'd_vec_y_{i_keys}'] = []
        dict_distance_vector[f'd_vec_z_{i_keys}'] = []
    return dict_distance_vector

def get_distance_by_vector(ele, indices, d, dict_distance_vector):
    """
    Populate the distance vector dictionary with x, y, z components for each element.

    Parameters:
    - ele (str): The element type (e.g., 'Cs_{d8r}').
    - indices (dict): A dictionary containing indices for the elements.
    - d (dict): A dictionary containing distance vectors.
    - dict_distance_vector (dict): A dictionary to store the distance vector components.

    Returns:
    - dict: Updated dictionary with x, y, and z components of the distance vectors.
    """
    for i_ele in range(len(indices[ele])):  # Iterate over each element
        dict_distance_vector[f'd_vec_x_{i_ele}'].append(d[i_ele][0][0])
        dict_distance_vector[f'd_vec_y_{i_ele}'].append(d[i_ele][0][1])
        dict_distance_vector[f'd_vec_z_{i_ele}'].append(d[i_ele][0][2])
    return dict_distance_vector

def get_cs_d8r_axial_lateral_dist_list_wet(indices, dist_dict_vector):
    """
    Separate axial and lateral distance components for Cs_{d8r} in wet conditions.

    Parameters:
    - indices (dict): A dictionary containing indices for Cs and d8r positions.
    - dist_dict_vector (dict): A dictionary containing the distance vector components.

    Returns:
    - list_axial: A list of axial distance components.
    - list_lateral: A list of lateral distance components.
    """
    list_axial = []
    list_lateral = []
    for i_cs in range(len(indices['Cs_{d8r}'])):
        if i_cs in [4, 5]:
            list_axial.append(dist_dict_vector[f'd_vec_x_{i_cs}'])
            list_lateral.extend([dist_dict_vector[f'd_vec_y_{i_cs}'], dist_dict_vector[f'd_vec_z_{i_cs}']])
        elif i_cs in [1, 3]:
            list_axial.append(dist_dict_vector[f'd_vec_y_{i_cs}'])
            list_lateral.extend([dist_dict_vector[f'd_vec_x_{i_cs}'], dist_dict_vector[f'd_vec_z_{i_cs}']])
        elif i_cs in [2, 0]:
            list_axial.append(dist_dict_vector[f'd_vec_z_{i_cs}'])
            list_lateral.extend([dist_dict_vector[f'd_vec_x_{i_cs}'], dist_dict_vector[f'd_vec_y_{i_cs}']])
    return list_axial, list_lateral

def threshold_percentage(threshold, data):
    """
    Calculate the percentage of data points above and below a given threshold.

    Parameters:
    - threshold (float): The threshold value.
    - data (list or np.array): The data to evaluate.

    Returns:
    - tuple: A tuple containing the percentage of data points above and below the threshold.
    """
    list_large = [i for i in data if i > threshold]
    list_small = [i for i in data if i <= threshold]
    large_percentage = len(list_large) / len(data)
    small_percentage = len(list_small) / len(data)
    return large_percentage, small_percentage
