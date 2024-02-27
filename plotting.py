#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By:       Tobias Meissner
# Created Date:     15.01.2024
# Date Modified:    27.02.2024
# Python Version:   3.8.5
# Dependencies:     NumPy (1.24.4), tensorflow (2.10.1), SciPy (1.10.1), vtk (9.3.0), matplotlib (3.7.2),
#                   Pandas (2.0.3), Seaborn (0.12.2), openCV (4.6.0), bottleneck (1.3.5)
# License:          GNU GENERAL PUBLIC LICENSE Version 3
#
# This script belongs to the publication "3D-Localization of Single Point-Like Gamma Sources with a Coded Aperture
# Camera" from Tobias Meissner, Laura Antonia Cerbone, Paolo Russo, Werner Nahm, and Juergen Hesser. More details can
# be found there.
# ----------------------------------------------------------------------------

import matplotlib.patheffects as path_effects
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def boxplots(dataframe, hue, title=""):
    df_melted = dataframe.melt(value_vars=["ErrorX", "ErrorY", "ErrorZ"], id_vars=hue, var_name='Variable',
                               value_name='Value')

    fig, ax = plt.subplots(1, 1, dpi=400, figsize=(8, 4.5))
    bp = sns.boxplot(df_melted, x='Variable', y='Value', hue=hue, showmeans=True, whis=(0, 100), ax=ax,
                     meanprops={"marker": "x",
                                "markeredgecolor": "black",
                                "markersize": "10"}, palette="Set2")
    for patch in ax.patches:
        fc = patch.get_facecolor()
        patch.set_facecolor(matplotlib.colors.to_rgba(fc, 0.5))

    sp = sns.stripplot(df_melted, x='Variable', y='Value', hue=hue, alpha=1.0, dodge=True, ax=ax, legend=None,
                       palette="Set2")

    add_median_labels(ax)
    ax.set_ylabel("(Estimated - GT) in mm")
    ax.set_xticks([0, 1, 2], ["X", "Y", "Z"])
    ax.set_xlabel("")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def add_median_labels(ax: plt.Axes, fmt: str = ".1f") -> None:
    """Add text labels to the median lines of a seaborn boxplot.

    Args:
        ax: plt.Axes, e.g. the return value of sns.boxplot()
        fmt: format string for the median value
    Stolen from https://stackoverflow.com/questions/52832767/how-to-create-upright-vertical-oriented-text-in-matplotlib
    """
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if "Patch" in str(c)]
    start = 4
    if not boxes:  # seaborn v0.13 => fill=False => no patches => +1 line
        boxes = [c for c in ax.get_lines() if len(c.get_xdata()) == 5]
        start += 1
    lines_per_box = len(lines) // len(boxes)
    for median in lines[start::lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if len(set(median.get_xdata())) == 1 else y
        text = ax.text(x + 0.08, y, f'{value:{fmt}}', ha='center', va='center',
                       fontweight='bold', color='white', rotation=90)
        # create median-colored border around white text for contrast
        text.set_path_effects([path_effects.Stroke(linewidth=3, foreground=median.get_color()),
                               path_effects.Normal()])


def plotp(data_cube, title="", xyres=None, zres=None, projection_fnc=np.sum, colorbar=True, subtitles=None):
    """
    :param subtitles: subtitles which go over the left and right sub-plots
    :param colorbar: if a colorbar should be depicted or not.
    :param data_cube: 3d numpy array to make projection figure of.
    :param title: str. Comes at the top of figure.
    :param xyres: in mm. Used for the ticks.
    :param zres: in mm. Used for the ticks.
    :param projection_fnc: Should be either np.sum or np.max. Different styles of projecting the cube to a 2D
    representation. It is somewhat hard to get the actual voxel values from np.sum.
    :return: A plt.figure which can then be saved, shown or closed.
    """
    data_cube = np.array(data_cube).squeeze()
    n = data_cube.shape[0]
    if xyres and zres:
        x_mm, y_mm, z_mm = data_cube.shape[0] * xyres, data_cube.shape[1] * xyres, data_cube.shape[2] * zres

    if subtitles is None:
        subtitles = ["Sum Projection along Z direction.", "Sum Projection along Y direction."]

    plt.close('all')
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    im = axs[0].imshow(projection_fnc(data_cube, 2))
    if colorbar:
        fig.colorbar(im, ax=axs[0])
    axs[0].set_title(subtitles[0])
    if xyres and zres:
        axs[0].set_xticks(np.linspace(0, n - 1, 6).astype(int), labels=np.linspace(0, y_mm, 6).astype(int))
        axs[0].set_yticks(np.linspace(0, n - 1, 6).astype(int), labels=np.linspace(0, x_mm, 6).astype(int))
    axs[0].set_xlabel("y [mm]")
    axs[0].set_ylabel("x [mm]")
    im = axs[1].imshow(projection_fnc(data_cube, 1))
    if colorbar:
        fig.colorbar(im, ax=axs[1])
    axs[1].set_title(subtitles[1])
    if xyres and zres:
        axs[1].set_xticks(np.linspace(0, n - 1, 6).astype(int), labels=np.linspace(0, z_mm, 6).astype(int))
        axs[1].set_yticks(np.linspace(0, n - 1, 6).astype(int), labels=np.linspace(0, x_mm, 6).astype(int))
    axs[1].set_xlabel("z [mm]")
    axs[1].set_ylabel("x [mm]")
    fig.suptitle(title)
    plt.tight_layout()
    # plt.show()
    return fig


def plot(img, title="", ticks=True, colorbar=True, cmap=None, clim=None, filename="", dpi=None, cb_format=None,
         fontsize=None):
    plt.clf()
    plt.cla()
    plt.imshow(np.squeeze(img), cmap=cmap)
    if ticks == False:
        plt.xticks([])
        plt.yticks([])
    if colorbar:
        cb = plt.colorbar(format=cb_format)
    if clim:
        plt.clim(clim)
    plt.title(str(title), fontsize=fontsize)

    if fontsize:
        cb.ax.tick_params(labelsize=fontsize)
    if dpi:
        plt.gcf().set_dpi(dpi)
    plt.tight_layout()
    fig = plt.gcf()
    if filename == "":
        plt.show()
    else:
        plt.savefig(filename)
    return fig
