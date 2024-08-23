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

def get_cs_s6r_dist_wet(atoms, indices):
    """Calculate distances between Cs at s6r positions and corresponding sites in wet conditions."""
    d = {}
    d[0] = atoms.get_distances(indices['Cs_{s6r}'][0], indices['s6r'][7], mic=True)
    d[1] = atoms.get_distances(indices['Cs_{s6r}'][1], indices['s6r'][3], mic=True)
    d[2] = atoms.get_distances(indices['Cs_{s6r}'][2], indices['s6r'][6], mic=True)
    d[3] = atoms.get_distances(indices['Cs_{s6r}'][3], indices['s6r'][2], mic=True)
    return d

def get_cs_d8r_dist_wet(atoms, indices):
    """Calculate distances between Cs at d8r positions and corresponding sites in wet conditions."""
    d = {}
    d[0] = atoms.get_distances(indices['Cs_{d8r}'][0], indices['d8r'][5], mic=True)
    d[1] = atoms.get_distances(indices['Cs_{d8r}'][1], indices['d8r'][3], mic=True)
    d[2] = atoms.get_distances(indices['Cs_{d8r}'][2], indices['d8r'][2], mic=True)
    d[3] = atoms.get_distances(indices['Cs_{d8r}'][3], indices['d8r'][0], mic=True)
    d[4] = atoms.get_distances(indices['Cs_{d8r}'][4], indices['d8r'][1], mic=True)
    d[5] = atoms.get_distances(indices['Cs_{d8r}'][5], indices['d8r'][4], mic=True)
    return d

def get_cs_d8r_dist_wet_vector(atoms, indices):
    """Calculate distance vectors between Cs at d8r positions and corresponding sites in wet conditions."""
    d = {}
    d[0] = atoms.get_distances(indices['d8r'][5], indices['Cs_{d8r}'][0], mic=True, vector=True)
    d[1] = atoms.get_distances(indices['d8r'][3], indices['Cs_{d8r}'][1], mic=True, vector=True)
    d[2] = atoms.get_distances(indices['d8r'][2], indices['Cs_{d8r}'][2], mic=True, vector=True)
    d[3] = atoms.get_distances(indices['d8r'][0], indices['Cs_{d8r}'][3], mic=True, vector=True)
    d[4] = atoms.get_distances(indices['d8r'][1], indices['Cs_{d8r}'][4], mic=True, vector=True)
    d[5] = atoms.get_distances(indices['d8r'][4], indices['Cs_{d8r}'][5], mic=True, vector=True)
    return d

def get_dict_distance_vector(ele, indices):
    """Initialize a dictionary to store distance vectors."""
    list_keys = []
    for i_keys in range(len(indices[ele])):
        list_keys.append(f'd_vec_x_{i_keys}')
        list_keys.append(f'd_vec_y_{i_keys}')
        list_keys.append(f'd_vec_z_{i_keys}')
    dict_distance_vector = {key: [] for key in list_keys}
    return dict_distance_vector

def get_distance_by_vector(ele, indices, d, dict_distance_vector):
    """Populate distance vector dictionary with x, y, z components."""
    for i_ele in range(len(indices[ele])):  # Iterate over all Cs_{d8r} atoms
        for i_xyz, axis in enumerate(['x', 'y', 'z']):
            dict_distance_vector[f'd_vec_{axis}_{i_ele}'].append(d[i_ele][0][i_xyz])
    return dict_distance_vector

def get_cs_d8r_axial_lateral_dist_list_wet(indices, dist_dict_vector):
    """Separate axial and lateral distance components for Cs_{d8r} in wet conditions."""
    list_axial = []
    list_lateral = []
    for i_cs in range(len(indices['Cs_{d8r}'])):
        if i_cs in [0, 5]:
            list_axial.append(dist_dict_vector[f'd_vec_x_{i_cs}'])
            list_lateral.extend([dist_dict_vector[f'd_vec_y_{i_cs}'], dist_dict_vector[f'd_vec_z_{i_cs}']])
        elif i_cs in [1, 4]:
            list_axial.append(dist_dict_vector[f'd_vec_y_{i_cs}'])
            list_lateral.extend([dist_dict_vector[f'd_vec_x_{i_cs}'], dist_dict_vector[f'd_vec_z_{i_cs}']])
        elif i_cs in [2, 3]:
            list_axial.append(dist_dict_vector[f'd_vec_z_{i_cs}'])
            list_lateral.extend([dist_dict_vector[f'd_vec_x_{i_cs}'], dist_dict_vector[f'd_vec_y_{i_cs}']])
    return list_axial, list_lateral

def threshold_percentage(threshold, data):
    """Calculate the percentage of data points above and below a given threshold."""
    list_large = [i for i in data if i > threshold]
    list_small = [i for i in data if i <= threshold]
    large_percentage = len(list_large) / len(data)
    small_percentage = len(list_small) / len(data)
    return large_percentage, small_percentage

def plot_2dhist_contour(xdata, ydata):
    """Plot a 2D histogram with contour using Plotly."""
    fig = go.Figure()

    # 2D contour histogram
    fig.add_trace(go.Histogram2dContour(
        x=xdata,
        y=ydata,
        colorscale='Blues',
        reversescale=True,
        xaxis='x',
        yaxis='y'
    ))

    # Scatter plot overlay
    fig.add_trace(go.Scatter(
        x=xdata,
        y=ydata,
        xaxis='x',
        yaxis='y',
        mode='markers',
        marker=dict(
            color='rgba(0,0,0,0.3)',
            size=3
        )
    ))

    # Marginal histograms
    fig.add_trace(go.Histogram(
        y=ydata,
        xaxis='x2',
        marker=dict(
            color='rgba(0,0,0,1)'
        )
    ))
    fig.add_trace(go.Histogram(
        x=xdata,
        yaxis='y2',
        marker=dict(
            color='rgba(0,0,0,1)'
        )
    ))

    # Update layout for figure
    fig.update_layout(
        autosize=False,
        xaxis=dict(
            zeroline=False,
            domain=[0, 0.85],
            showgrid=False
        ),
        yaxis=dict(
            zeroline=False,
            domain=[0, 0.85],
            showgrid=False
        ),
        xaxis2=dict(
            zeroline=False,
            domain=[0.85, 1],
            showgrid=False
        ),
        yaxis2=dict(
            zeroline=False,
            domain=[0.85, 1],
            showgrid=False
        ),
        height=600,
        width=600,
        bargap=0,
        hovermode='closest',
        showlegend=False
    )

    return fig

