# date: Oct 24, 2023
# @author:Kun-Lin Wu

import sys, os
from md_utils import *
from md_tools import *
from ase import io
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas
import plotly.express as px
import plotly.graph_objects as go
from ase.spacegroup import crystal
from md_helper import get_indices, dict_to_atoms
from ase import neighborlist
from ase import Atoms
from scipy.stats import norm, alpha
from ase.visualize import view
from ase.build import molecule
from scipy.optimize import curve_fit



def get_cs_s6r_dist_wet(atoms, indices):
    d = {}
    d[0] = atoms.get_distances(indices['Cs_{s6r}'][0], indices['s6r'][7], mic=True)
    d[1] = atoms.get_distances(indices['Cs_{s6r}'][1], indices['s6r'][3], mic=True)
    d[2] = atoms.get_distances(indices['Cs_{s6r}'][2], indices['s6r'][6], mic=True)
    d[3] = atoms.get_distances(indices['Cs_{s6r}'][3], indices['s6r'][2], mic=True)

    return d


def get_cs_d8r_dist_wet(atoms, indices):
    d = {}
    d[0] = atoms.get_distances(indices['Cs_{d8r}'][0], indices['d8r'][5],  mic=True)
    d[1] = atoms.get_distances(indices['Cs_{d8r}'][1], indices['d8r'][3],  mic=True)
    d[2] = atoms.get_distances(indices['Cs_{d8r}'][2], indices['d8r'][2],  mic=True)
    d[3] = atoms.get_distances(indices['Cs_{d8r}'][3], indices['d8r'][0],  mic=True)
    d[4] = atoms.get_distances(indices['Cs_{d8r}'][4], indices['d8r'][1],  mic=True)
    d[5] = atoms.get_distances(indices['Cs_{d8r}'][5], indices['d8r'][4],  mic=True)

    return d


def get_cs_d8r_dist_wet_vector(atoms, indices):
    d = {}
    d[0] = atoms.get_distances(indices['d8r'][5], indices['Cs_{d8r}'][0], mic=True, vector=True)
    d[1] = atoms.get_distances(indices['d8r'][3], indices['Cs_{d8r}'][1], mic=True, vector=True)
    d[2] = atoms.get_distances(indices['d8r'][2], indices['Cs_{d8r}'][2], mic=True, vector=True)
    d[3] = atoms.get_distances(indices['d8r'][0], indices['Cs_{d8r}'][3], mic=True, vector=True)
    d[4] = atoms.get_distances(indices['d8r'][1], indices['Cs_{d8r}'][4], mic=True, vector=True)
    d[5] = atoms.get_distances(indices['d8r'][4], indices['Cs_{d8r}'][5], mic=True, vector=True)

    return d


def get_dict_distance_vector(ele, indices):
    list_keys = []
    for i_keys in range(len(indices[ele])):
        list_keys.append('d_vec_x_%s' % i_keys)
        list_keys.append('d_vec_y_%s' % i_keys)
        list_keys.append('d_vec_z_%s' % i_keys)

    dict_distance_vector = {key: [] for key in list_keys}

    return dict_distance_vector


def get_distance_by_vector(ele, indices, d, dict_distance_vector):
    for i_ele in range(len(indices[ele])):  # there are 6 cs_d8r
        for i_xyz in ['x', 'y', 'z']:
            if i_xyz == 'x':
                dict_distance_vector['d_vec_%s_%s' % (i_xyz, i_ele)].append(d[i_ele][0][0])
            if i_xyz == 'y':
                dict_distance_vector['d_vec_%s_%s' % (i_xyz, i_ele)].append(d[i_ele][0][1])
            if i_xyz == 'z':
                dict_distance_vector['d_vec_%s_%s' % (i_xyz, i_ele)].append(d[i_ele][0][2])

    return dict_distance_vector


def get_cs_d8r_axial_lateral_dist_list_wet(indices, dist_dict_vector):
    '''
    get the axial direction of each cs_d8r atoms. You can think of a unit cell in cube structure and
    there are 6 sides with 2 planes each pointing to each xyz directions.
    '''

    list_axial = []
    list_lateral = []
    for i_cs in range(len(indices['Cs_{d8r}'])):
        if i_cs == 0 or i_cs == 5:
            list_axial.append(dist_dict_vector['d_vec_x_%s' % i_cs])
            list_lateral.append(dist_dict_vector['d_vec_y_%s' % i_cs])
            list_lateral.append(dist_dict_vector['d_vec_z_%s' % i_cs])
        if i_cs == 1 or i_cs == 4:
            list_axial.append(dist_dict_vector['d_vec_y_%s' % i_cs])
            list_lateral.append(dist_dict_vector['d_vec_x_%s' % i_cs])
            list_lateral.append(dist_dict_vector['d_vec_z_%s' % i_cs])
        if i_cs == 2 or i_cs == 3:
            list_axial.append(dist_dict_vector['d_vec_z_%s' % i_cs])
            list_lateral.append(dist_dict_vector['d_vec_x_%s' % i_cs])
            list_lateral.append(dist_dict_vector['d_vec_y_%s' % i_cs])

    return list_axial, list_lateral


def threshold_percentage(threshold, data):
    list_large = []
    list_small = []
    for i_data in data:
        if i_data > threshold:
            list_large.append(i_data)
        else:
            list_small.append(i_data)
    large_percentage = len(list_large) / len(data)
    small_percentage = len(list_small) / len(data)

    return large_percentage, small_percentage


def plot_2dhist_contour(xdata, ydata):
    '''
    plot the 2d histogram with contour using plotly
    '''

    fig = go.Figure()
    fig.add_trace(go.Histogram2dContour(
        x=xdata,
        y=ydata,
        colorscale='Blues',
        reversescale=True,
        xaxis='x',
        yaxis='y'
    ))
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