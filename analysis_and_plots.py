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

import numpy as np
import pandas as pd
from argparse import Namespace
import matplotlib.pyplot as plt

from localize import localize_all_sources
from plotting import plot, plotp, boxplots


p = dict()
p["files"] = r"./3d_reconstructions/experimental/x*_d???.??.tif"
p["files"] = r"./3d_reconstructions/mc_simulation/x*_MC_*d???.??.tif"
# Geometry parameters:
p["b"] = 20.0
p["hd"] = 14.08
p["pixels"] = 256
p["r"] = 0.08 / 2.0
p["hm"] = 9.92
p["t"] = 0.11
# True for generating informational plots
p["debug"] = False
# Either localize source with the Iterative source localization method ("ISL") or the center of mass method ("COM")
# ~~~~~~~~~~~~~~ ISL specific parameters ~~~~~~~~~~~~~~
p["method"] = "ICL"
# Type of intial guess for the ISL algorithm: Either "true", "random", or "max"
p["IG_type"] = "true"
# Function to fit: Either "EMG", "gauss" or "max"
p["fit"] = "EMG"
# ROI diameter in mm: For the 241Am source, we use 0.65mm (nominal diameter 1.0mm)
p["ROId_mm"] = 0.65
# ~~~~~~~~~~~~~~ Center of mass specific parameters ~~~~~~~~~~~~~~
p["method"] = "COM"
p["fit"] = "None"
p["ROId_mm"] = np.NAN


# +++++++++++++++++++++++++++++++++++++++++++++++++ EMG vs. Gauss +++++++++++++++++++++++++++++++++++++++++++++++++
"""     """
p["method"] = "ISL"
p["fit"] = "EMG"
p["ROId_mm"] = 0.65
p["debug"] = False
c = localize_all_sources(Namespace(**p))

p["fit"] = "gauss"
c_gauss = localize_all_sources(Namespace(**p))

df = pd.concat([c, c_gauss])

fig = boxplots(df, "Fit", "Localization error vs. fitting function")
fig.show()
plt.close(fig)

# ++++++++++++++++++++++++++++++++++++++++++++++ Sensitivity analysis IG ++++++++++++++++++++++++++++++++++++++++++++++
"""     """
p["fit"] = "EMG"
p["ROId_mm"] = 0.65
p["method"] = "ISL"
p["IG_type"] = "true"
p["debug"] = False
p["files"] = r"./3d_reconstructions/mc_simulation/x00*_MC_*d???.??.tif"
c_true_IG = localize_all_sources(Namespace(**p))

p["IG_type"] = "random5"
c_random5_0 = pd.DataFrame()
for _ in range(5):
    temp = localize_all_sources(Namespace(**p))
    c_random5_0 = pd.concat([c_random5_0, temp])

p["IG_type"] = "random10"
c_random10_0 = pd.DataFrame()
for _ in range(5):
    temp = localize_all_sources(Namespace(**p))
    c_random10_0 = pd.concat([c_random10_0, temp])

p["IG_type"] = "random15"
c_random15_0 = pd.DataFrame()
for _ in range(5):
    temp = localize_all_sources(Namespace(**p))
    c_random15_0 = pd.concat([c_random15_0, temp])

p["IG_type"] = "max"
c_max_IG = localize_all_sources(Namespace(**p))

df = pd.concat([c_true_IG, c_random5_0, c_random10_0, c_random15_0, c_max_IG])

fig = boxplots(df, "IG", "Localization error vs. initial guess")
fig.show()
plt.close(fig)


# +++++++++++++++++++++++++++++++++++++++++++++++ ISL vs. COM on TOPAS +++++++++++++++++++++++++++++++++++++++++++++++
"""
"""
p["method"] = "ISL"
p["fit"] = "EMG"
p["IG_type"] = "true"
p["ROId_mm"] = 0.65
c_isl = localize_all_sources(Namespace(**p))
c_isl["Method"] = "ISL-EMG"

p["method"] = "COM"
p["fit"] = "None"
p["ROId_mm"] = np.NAN
c_com = localize_all_sources(Namespace(**p))
c_com["Method"] = "COM"

df = pd.concat([c_isl, c_com])

fig = boxplots(df, "Method", "Localization error: ISL-EMG vs. COM")
fig.show()
plt.close(fig)


# ++++++++++++++++++++++++++++++++++++++++++++++ TOPAS vs. Experimental ++++++++++++++++++++++++++++++++++++++++++++++
p["method"] = "ISL"
p["fit"] = "EMG"
p["IG_type"] = "true"
p["ROId_mm"] = 0.65
p["debug"] = False
p["files"] = r"./3d_reconstructions/mc_simulation/x*_MC_*d???.??.tif"
c_topas = localize_all_sources(Namespace(**p))
c_topas["Data"] = "TOPAS"

p["files"] = r"./3d_reconstructions/experimental/x*_d???.??.tif"
c_exp = localize_all_sources(Namespace(**p))
c_exp["Data"] = "Experimental"

df = pd.concat([c_topas, c_exp])

fig = boxplots(df, "Data", "TOPAS vs. Experimental data")
fig.show()
plt.close(fig)

# +++++++++++++++++++++++++++++++++ COM vs. ISL-Gaussian vs. ISL-EMG vs. ISL-EMG Exp +++++++++++++++++++++++++++++++++

p["files"] = r"./3d_reconstructions/mc_simulation/x*_MC_*d???.??.tif"
p["method"] = "COM"
p["fit"] = "None"
p["ROId_mm"] = np.NAN
c_1 = localize_all_sources(Namespace(**p))
c_1["Method"] = "COM (Sim.)"


p["method"] = "ISL"
p["fit"] = "gauss"
p["IG_type"] = "true"
p["ROId_mm"] = 0.65
c_2 = localize_all_sources(Namespace(**p))
c_2["Method"] = "ISL-Gaussian (Sim.)"

p["method"] = "ISL"
p["fit"] = "EMG"
p["IG_type"] = "true"
p["ROId_mm"] = 0.65
c_3 = localize_all_sources(Namespace(**p))
c_3["Method"] = "ISL-EMG (Sim.)"


p["files"] = r"./3d_reconstructions/experimental/x*_d???.??.tif"
p["method"] = "ISL"
p["fit"] = "EMG"
p["IG_type"] = "true"
p["ROId_mm"] = 0.65
c_4 = localize_all_sources(Namespace(**p))
c_4["Method"] = "ISL-EMG (Exp.)"

df = pd.concat([c_1, c_2, c_3, c_4])

fig = boxplots(df, "Method", "")
fig.show()
plt.close(fig)

