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

import time
import tifffile
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.stats import exponnorm, norm

import tensorflow as tf
from cv2 import circle as cv2circle
from bottleneck import nanmean as bn_nanmean

from plotting import plot, plotp


def px2mm(pos_px, shape, fov):
    """
    pos_px: (row, column), shape: (row, colum), fov: float value in mm
    based on a central coordinate system.
    """
    lat_pos_rel = (np.array(pos_px)) / (np.array(shape) - 1) - 0.5
    if np.any(np.logical_or(np.less(lat_pos_rel, -0.5), np.greater(lat_pos_rel, 0.5))):
        print("px2mm: ", lat_pos_rel)
    pos_mm = fov * lat_pos_rel
    return pos_mm


def mm2px(pos_mm, shape, fov):
    """ Returns NAN array, if the position is outside the image. """
    pos_rel = (np.array(pos_mm) / fov) + 0.5
    # Should be within [0, 1]:
    if np.any(np.logical_or(np.less(pos_rel, 0.0), np.greater(pos_rel, 1.0))):
        return np.NAN * np.zeros_like(pos_rel)
    temp = pos_rel * (np.array(shape) - 1.0)
    pos_px = np.round(temp).astype(int)
    return pos_px


def get_accorsi_fov(a, b, mt, d):
    """
    This equation is straight out of Fujii2012.

    a: distance between source and mask
    b: distance between mask and detector
    mt: mask thickness in mm
    d: height of central mask pattern
    """
    # If mask length is smaller than central pattern projection:
    if a < b - mt:
        # This line is different from get_accorsi_fov_wrong
        afov = (a + b) / (b - mt / 2) * d
    # Else:
    else:
        afov = (a + b) * (a + 3 / 2 * mt) / ((a + mt / 2) * (b + mt / 2)) * d
    return afov


def zkey_float(k):
    """ returns the float32 value between the last 'z' and '_Minipix' where "p" denotes the decimal point.
    In our case, that should be the distance in mm."""
    try:
        str_val = k[k.rfind("z") + 1:k.find("_Minipix")]
        str_val = str_val.replace("p", ".")
        return_val = np.float32(float(str_val))
    except:
        print("z_key_float: Could not find z-value in string: %s" % k)
        return_val = -1.0
    return return_val


def r2(func, p, x_vals, y_vals):
    """ Calculate the coefficient of determination (R²) of a function func with its fitted parameters p."""
    res = y_vals - func(x_vals, *p)
    ss_res = np.sum(res ** 2)
    ss_tot = np.sum((y_vals - np.mean(y_vals)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared


def find_max_pos(img, probe_radius, restrain_to_center=False, restrain_anchor=None, return_signal=False):
    """
    Returns the 2D indices of the pixel, where the average intensity within a circle of given radius is maximal.
    NaNs are ignored.
    restrain_anchor: must be given in (row, column)
    """
    if restrain_anchor is None:
        restrain_anchor = img.shape[0] // 2, img.shape[1] // 2
    STRIDE = 1

    # Generate circular probe kernel:
    circ_kernel = np.zeros((probe_radius * 2 + 1, probe_radius * 2 + 1), np.float32)
    circ_kernel = cv2circle(circ_kernel, (circ_kernel.shape[0] // 2, circ_kernel.shape[1] // 2),
                            radius=probe_radius, thickness=-1, color=1.0)
    circ_kernel = circ_kernel / np.sum(circ_kernel)
    circ_kernel = circ_kernel[..., None, None]

    # Calculate the mean value of each circle:
    # IMPORTANT: Unlike in cnr_via_conv, here "SAME" padding is used!
    E_x = tf.nn.conv2d(img[None, ..., None], circ_kernel, padding="VALID", strides=STRIDE)
    E_x = np.squeeze(E_x)
    E_x = np.pad(E_x, [[probe_radius, probe_radius], [probe_radius, probe_radius]], constant_values=np.NAN)

    if restrain_to_center:
        mask = np.ones_like(E_x) * np.NAN
        # Switch order here for cv2:
        mask = cv2circle(mask, [restrain_anchor[1], restrain_anchor[0]], radius=E_x.shape[0] // 4, thickness=-1,
                         color=1.0)
        E_x = (E_x * mask)

    try:
        i_max = np.nanargmax(E_x.flatten())
        i_max_2d = np.unravel_index(i_max, np.shape(E_x))
    except:
        i_max_2d = np.NAN

    if return_signal:
        return i_max_2d, np.nanmax(E_x)
    return i_max_2d


def calc_determ_cnr(img, probe_radius, source_pos):
    """
    Calculates the CNR similar to our "Assessment of the axial resolution...". Difference here, that this is
    calculated with convolutions and numpy, not ImageJ.
    The source position should be fined first with find_max_pos(.).
    source_pos is (row, column).
    """
    source_pos = np.squeeze(source_pos)
    assert np.size(source_pos) == 2, "Given source position must only contain two indices. Given: " + str(source_pos)

    # If source position contains NAN, NAN should be returned:
    if np.any(np.isnan(source_pos)):
        return np.NAN, np.NAN, np.NAN, np.NAN,

    # In order to not alter the original passed-by-reference image:
    img = img.copy()

    # DO NOT DARE TO ALTER THE STRIDE PARAMETER! OTHERWISE, THE DELETION FOR THE OVERLAPPING
    # REGION WILL NOT WORK!
    STRIDE = 1
    # Generate circular probe kernel:
    circ_kernel = np.zeros((probe_radius * 2 + 1, probe_radius * 2 + 1), np.float32)
    circ_kernel = cv2circle(circ_kernel, (circ_kernel.shape[0] // 2, circ_kernel.shape[1] // 2),
                            radius=probe_radius, thickness=-1, color=1.0)
    circ_kernel = circ_kernel / np.sum(circ_kernel)
    circ_kernel = circ_kernel[..., None, None]

    # SAME padding to keep the coordinates.
    # Done: VALID padding plus padding with NANs to get back to same size?!?!
    E_x = tf.nn.conv2d(img[None, ..., None], circ_kernel, padding="VALID", strides=STRIDE)
    E_x = np.squeeze(E_x)
    E_x = np.pad(E_x, [[probe_radius, probe_radius], [probe_radius, probe_radius]], constant_values=np.NAN)
    assert np.shape(E_x) == np.shape(img), "Wrong shape here: Something went wrong with the padding after convolution"

    # Calculate the mean of the elementwise-squared image:
    # Done: VALID padding plus padding with NANs to get back to same size?!?!
    E_x2 = tf.nn.conv2d((img ** 2)[None, ..., None], circ_kernel, padding="VALID", strides=STRIDE)
    E_x2 = np.squeeze(E_x2)
    E_x2 = np.pad(E_x2, [[probe_radius, probe_radius], [probe_radius, probe_radius]], constant_values=np.NAN)
    assert np.shape(E_x2) == np.shape(img), "Wrong shape here: Something went wrong with the padding after convolution"

    # Var(X) = E(X²) - E(X)²
    Var_x = E_x2 - E_x ** 2
    # Due to the subtraction, tiny negative values can occur. We replace them with zeros.
    Var_x[Var_x < 0] = 0.0
    Std_x = np.sqrt(Var_x)

    # The signal voxel is already decided and now we only delete the neighboring circles:
    # Store mean value of Signal ROI:
    signal_mean = E_x[source_pos[0], source_pos[1]]

    # Delete the ROIs which are
    E_x = cv2circle(E_x, [source_pos[1], source_pos[0]],
                    radius=2 * probe_radius, thickness=-1, color=np.NAN)
    Std_x = cv2circle(Std_x, [source_pos[1], source_pos[0]],
                      radius=2 * probe_radius, thickness=-1, color=np.NAN)

    b = bn_nanmean(E_x.flatten())
    sigma = bn_nanmean(Std_x.flatten())
    # Final CNR is the difference of the signal to the mean of all other probes
    # divided by the mean of all standard deviations
    cnr = (signal_mean - b) / sigma

    return cnr, signal_mean, b, sigma


def isl(imgs, dists, fovs, source_diameter_mm, z_guess_mm, tofit="EMG", debug=False, return_r2=False):
    """
    Estimates the 3D position of a 3D reconstructions 'imgs', based on the given field of view per image ('fovs'),
    the source diameter in mm ('source_diameter_mm') and an initial guess for the z-component ('z_guess_mm').
    'fofit' can be either EMG, Gauss or Max and determines which way of estimating the z-component is used.
    """

    assert len(imgs) == len(dists) == len(fovs), (f"Number of images, distances and FOVs does not "
                                                  f"match: {len(imgs)}, {len(dists)}, {len(fovs)}")
    rint = lambda z: np.round(z).astype(int)

    exponorm2fit = lambda z, a, b, loc, scale, lamb: (a + (b - a) * (np.sqrt(2 * np.pi) * scale) *
                                                      exponnorm.pdf(z, K=1 / (scale * lamb), loc=loc, scale=scale))
    gauss2fit = lambda z, a, b, loc, scale: (a + (b - a) * (np.sqrt(2 * np.pi) * scale) *
                                             norm.pdf(z, loc=loc, scale=scale))


    def fit_expnorm(cnrs, dists):
        """ Fitting, and calculating R² and the mode for the exponentially modified Gaussian."""
        nanmask = ~np.isnan(cnrs)
        p1s, pconv_expnorm = curve_fit(exponorm2fit, dists[nanmask], cnrs[nanmask],
                                        p0=[np.nanmin(cnrs), np.nanmax(cnrs), dists[np.nanargmax(cnrs)], 1, 1],
                                        bounds=[[-np.inf, -np.inf, 0, 0, -np.inf],
                                                [np.inf, np.inf, np.inf, np.inf, np.inf]], max_nfev=100_000)
        # Find the mode of the PDF quick and dirty!
        z_temp = np.mgrid[dists.min():dists.max():0.01]
        fit_temp = exponorm2fit(z_temp, *p1s)
        mode = z_temp[fit_temp == fit_temp.max()].mean()
        r2s = r2(exponorm2fit, p1s, dists[nanmask], cnrs[nanmask])
        return p1s, pconv_expnorm, r2s, mode

    def fit_gauss(cnrs, dists):
        """ Fitting, and calculating R² and the mode for the normal Gaussian. """
        nanmask = ~np.isnan(cnrs)
        p1s, pconv_norm = curve_fit(gauss2fit, dists[nanmask], cnrs[nanmask],
                                        p0=[np.nanmin(cnrs), np.nanmax(cnrs), dists[np.nanargmax(cnrs)], 1],
                                        bounds=[[-np.inf, -np.inf, 0, 0],
                                                [np.inf, np.inf, np.inf, np.inf]], max_nfev=100_000)
        # Find the mode of the PDF quick and dirty!
        z_temp = np.mgrid[dists.min():dists.max():0.01]
        fit_temp = gauss2fit(z_temp, *p1s)
        mode = z_temp[fit_temp == fit_temp.max()].mean()
        r2s = r2(gauss2fit, p1s, dists[nanmask], cnrs[nanmask])
        return p1s, pconv_norm, r2s, mode

    # Set some constants:
    T0 = time.time()
    MAX_ITERS = 10
    SOURCE_DIAMETER_MM = source_diameter_mm

    INITIAL_GUESS_MM = np.array(z_guess_mm)
    # Convert that to px now:
    z_ig_px = np.argmin((dists - INITIAL_GUESS_MM) ** 2)
    INITIAL_GUESS_PX = z_ig_px

    # Calculate ROI from mm to pixels:
    res_at_slice_mm = fovs[INITIAL_GUESS_PX] / np.shape(imgs[INITIAL_GUESS_PX])[0]
    ROI_r = rint(SOURCE_DIAMETER_MM / res_at_slice_mm / 2.0)

    pos_px_hist = [[np.NAN, np.NAN, INITIAL_GUESS_PX], ]
    pos_mm_hist = [[np.NAN, np.NAN, INITIAL_GUESS_MM], ]
    pos_cnr_hist = [np.NAN, ]
    new_pos_px, old_pos_px = pos_px_hist[0], False
    new_pos_mm = pos_mm_hist[0]
    i = 0
    success = False
    while (i < MAX_ITERS) and (success is not True):
        # Old position becomes new position for comparison later:
        old_pos_px = new_pos_px

        # Find lateral source position in currently selected slice:
        lat_pos_px = find_max_pos(imgs[new_pos_px[2]], probe_radius=ROI_r, restrain_to_center=False)
        if lat_pos_px is np.NAN:
            success = False
            break

        # Convert pixel position to mm
        lat_pos_mm = px2mm(lat_pos_px[0:2], np.shape(imgs[new_pos_px[2]]), fovs[new_pos_px[2]])

        cnrs_pack = [calc_determ_cnr(t, ROI_r, lat_pos_px) for t in imgs]
        (cnrs, signal_means, background_means, background_sigmas) = np.transpose(cnrs_pack)

        # New z-slice to search for the highest CNR is the one where this CNR profile peaks:
        this_z_idx = np.nanargmax(cnrs)
        new_pos_px = lat_pos_px + (this_z_idx,)
        new_pos_mm = np.hstack([lat_pos_mm, dists[new_pos_px[2]]])

        # Save results for later:
        pos_px_hist.append(new_pos_px)
        pos_mm_hist.append(new_pos_mm)
        pos_cnr_hist.append(np.nanmax(cnrs))

        # Stop when the position does not change anymore:
        if new_pos_px == old_pos_px:
            success = True
            break

        # Calculate the new ROI radius:
        res_at_slice_mm = fovs[new_pos_px[2]] / np.shape(imgs[new_pos_px[2]])[0]
        ROI_r = rint(SOURCE_DIAMETER_MM / res_at_slice_mm / 2.0)

        # Increment iteration counter:
        i += 1

    # Fit expo-norm PDF into CNR, if previous procedure was successfully:
    if success is True:
        if tofit.lower() == "emg":
            p1s, _, r2s, mode = fit_expnorm(cnrs, dists)
        elif tofit.lower() == "gauss":
            p1s, _, r2s, mode = fit_gauss(cnrs, dists)
        elif tofit.lower() == "max":
            # Mean position of where CNR profile is maximum. More robust when multiple CNRs are maximum:
            p1s, r2s, mode = (np.NAN,), np.nan, np.mean(dists[cnrs == np.nanmax(cnrs)])

        # Update source position with the mode of the fitted curve:
        new_pos_mm[2] = mode

        # print("Resolution at %.2f mm distance is %.2fmm" % (dists[new_pos_px[2]], temp_res))
        # print("Source at has the following center: (%.2f, %.2f %.2f±???)mm (R²=%.2f)" % (*new_pos_mm[0:2], mode, r2s))

        # Debug plotting:
        if debug:
            from scipy.interpolate import interp1d
            z_temp = np.mgrid[dists.min():dists.max():0.01]
            if tofit.lower() == "emg":
                fit_temp = exponorm2fit(z_temp, *p1s)
            elif tofit.lower() == "gauss":
                fit_temp = gauss2fit(z_temp, *p1s)
            elif tofit.lower() == "max":
                fit_temp = interp1d(dists, cnrs, kind="nearest")(z_temp)

            f, ax = plt.subplots(dpi=500, figsize=(7, 7))
            ax.plot(z_temp, interp1d(dists, cnrs, kind="nearest")(z_temp), color="red", lw=3, zorder=1, label="CNR profile")
            ax.plot(z_temp, fit_temp, color="black", ls="--", dash_capstyle='round',
                    lw=3, label="%s fit" % tofit, zorder=3)
            old_view = ax._get_view()
            ax.vlines(mode, -1000, fit_temp.max(), lw=3, ls="--", color="k", capstyle='round')
            ax._set_view(old_view)
            ax.set_xlabel("Distance in mm", fontsize=18)
            ax.set_ylabel("CNR", fontsize=18)
            ax.grid(True)
            ax.set_axisbelow(True)
            ax.legend(fontsize=18)
            ax.tick_params(axis='both', which='major', labelsize=18)
            ax.tick_params(axis='both', which='minor', labelsize=14)
            f.tight_layout()
            f.suptitle("%s fit: Source has following center: (%.2f, %.2f %.2f)mm (R²=%.2f)\np=[%s]" %
                       (tofit, *new_pos_mm[0:2], mode, r2s, ', '.join('{:.2f}'.format(x) for x in p1s)),
                       fontsize="small")
            f.tight_layout()
            f.show()
            plt.close(f)

    else:
        r2s = np.NAN
        new_pos_mm = [np.NAN, np.NAN, np.NAN]

    print(f"ISL: {i+1} iterations, {(time.time() - T0):.2f}s. Converged? {success}. Fit? R²={r2s:.2f}."
          f" => ({new_pos_mm[0]:.2f}, {new_pos_mm[1]:.2f}, {new_pos_mm[2]:.2f})mm")

    if return_r2:
        return new_pos_mm, r2s
    return new_pos_mm


def com_vtk(imgs, dists, fovs, debug=False, debug_fn=""):
    """
    Similar to ISL, this methods estimates the 3D position based on the center of mass.
    """
    from vtk_funcs import tmCurviStructuredGrid, tmThreshold, tmLargestConnectedRegion, tmExtractPointsAndData
    from vtk_funcs import tmCleanUnstructuredGrid, getAccorsiGrid
    from vtk_funcs import tmSave

    assert len(imgs) == len(dists) == len(fovs), (f"Number of images, distances and FOVs does not "
                                                  f"match: {len(imgs)}, {len(dists)}, {len(fovs)}")
    # Switch to a [X, Y, Z] representation:
    imgs = np.transpose(imgs, [1, 2, 0])
    nx, ny, nz = np.shape(imgs)

    # Threshold
    imgs2p = imgs.copy()
    p99p9 = np.percentile(imgs2p.flatten(), 100 - 0.1)
    imgs2p[imgs2p < p99p9] = 0.0

    # Set some constants:
    T0 = time.time()

    # Generate grid for point cloud data:
    XX, YY, ZZ = getAccorsiGrid(dims=(nx, ny, nz), fovs=fovs, dists=dists)
    struct_vtk = tmCurviStructuredGrid(XX, YY, ZZ, pointData={"imgs2p": imgs2p.copy()})

    # This thresholding is good for getting rid of all the NANs AND the zeros from the thresholding above!
    thr_vtk = tmThreshold(struct_vtk, "imgs2p", lb=1e-8, ub=np.inf)
    connected_region = tmLargestConnectedRegion(thr_vtk)

    # Throw away all points that are not connected to the largest region:
    cleaned_region = tmCleanUnstructuredGrid(connected_region)

    # Only compute a CoM when thec cleaned & extracted region is not empty:
    if cleaned_region.GetPoints() is not None:

        pts, intensity = tmExtractPointsAndData(cleaned_region)
        x_cr, y_cr, z_cr = np.split(pts, 3, axis=1)

        # Some formatting and casting stuff:
        intensity = intensity.astype(np.float32)
        x_cr, y_cr, z_cr = x_cr.squeeze(), y_cr.squeeze(), z_cr.squeeze()

        # Determine centroid of first image half:
        m000 = np.sum(intensity)
        m100 = np.sum(intensity * x_cr)
        m010 = np.sum(intensity * y_cr)
        m001 = np.sum(intensity * z_cr)

        # Switch x, and y:
        com = np.array([m010, m100, m001], np.float32) / m000
    else:
        com = [np.NAN, np.NAN, np.NAN]

    if debug:
        com_px = mm2px(com[0:2], (nx, ny), fovs[np.argmin((com[2] - dists) ** 2)])
        com_px = np.hstack([com_px, dists[np.argmin((com[2] - dists) ** 2)]])

        h, _ = np.histogramdd(pts, range=((XX.min(), XX.max()), (YY.min(), YY.max()), (ZZ.min(), ZZ.max())),
                              bins=(nx, ny, nz))

        f = plotp(imgs2p, "NANMean projection imgs2p", xyres=1, zres=1, projection_fnc=np.nanmean)
        f.axes[0].annotate("x", (com_px[1], com_px[0]), color="red", fontsize=30, ha="center", va="center")
        f.axes[1].annotate("x", (com_px[2], com_px[0]), color="red", fontsize=30, ha="center", va="center")
        f.show()

        plotp(h, "Thresholded and LCR (cubicled)").show()

        plt.plot(dists, imgs2p.sum((0, 1))/m000, label="z-projection: only thresholded")
        plt.plot(dists, h.sum((0, 1))/h.sum(), label="z-projection: thresholded + LCR")
        plt.vlines(com[2], 0, (imgs2p.sum((0, 1))/m000).max(), color="r", ls="--", label="")
        plt.legend()
        plt.ylabel("normalized intensity")
        plt.xlabel("Distance in mm")
        plt.tight_layout()
        plt.show()

        tmSave(debug_fn + ".vts", struct_vtk)

    print(f"COM. 0.1% threshold: {p99p9:.2f}: {(time.time() - T0):.2f}s => ({com[0]:.2f}, {com[1]:.2f}, {com[2]:.2f})mm")
    return com


def localize_all_sources(p):
    # The source's nominal diameter is 1mm, and its FWHM was measured to be 0.65mm:
    SOURCE_FWHM_MM = p.ROId_mm

    # Collect all reconstruction file paths:
    fs = glob(p.files)

    fs = [f.replace("\\", "/") for f in fs]
    fs = sorted(fs, key=zkey_float)

    # Load files:
    imgs_in, img = [], None
    for f in fs:
        if f.endswith(".npy"):
            img = np.load(f)
        elif f.endswith(".tif") or f.endswith(".tiff"):
            img = tifffile.imread(f)
        img = np.squeeze(img).astype(np.float32)
        imgs_in.append(img)
    print("Loaded %i file(s)." % len(imgs_in))

    # Extract information from the file names
    gt_xs, gt_ys, gt_zs = [], [], []
    for f in fs:
        # Cut to file name:
        f = f.split("/")[-1]
        # Extract x, y and z coordinates from string
        this_x = float(f[f.find("x") + 1:f.rfind("y")])
        this_y = float(f[f.find("y") + 1:f.rfind("z")])
        gt_zs.append(zkey_float(f))
        gt_xs.append(this_x)
        gt_ys.append(this_y)

    # Collect mask-to-source distance and reconstruction method:
    reconstructed_at = [float(f[f.rfind("d") + 1: f.rfind(".tif")]) for f in fs]
    rec_method = []
    for f in fs:
        if "MURADecoded" in f:
            rec_method.append("MURADecoded")

    # Assemble dataframe and delete unnecessary variables
    df = pd.DataFrame({"Image": imgs_in, "Filename": fs, "GTx": gt_xs, "GTY": gt_ys, "GTZ": gt_zs,
                       "ReconstructionMethod": rec_method, "ReconstructedAt": reconstructed_at})
    del imgs_in, fs, gt_zs, rec_method, reconstructed_at

    est_pos = []
    gts = []
    r2s = []

    filtered_df = df[df["ReconstructionMethod"] == "MURADecoded"]
    for z in np.unique(filtered_df["GTZ"].to_numpy()):
        for y in np.unique(filtered_df["GTY"].to_numpy()):
            for x in np.unique(filtered_df["GTx"].to_numpy()):
                # Mask for the images at the right x, y, z position:
                mask = (filtered_df["GTx"] == x) & (filtered_df["GTY"] == y) & (filtered_df["GTZ"] == z)
                temp_df = filtered_df[mask]
                if temp_df.empty:
                    continue

                imgs = temp_df["Image"].to_numpy()
                dists = temp_df["ReconstructedAt"].to_numpy()
                this_f = np.unique(temp_df["Filename"])[0].split("/")[-1]

                # We resize all images to the same size by bilinear interpolation:
                max_shape = max([i.shape[0] for i in imgs])
                imgs = [np.squeeze(tf.image.resize(t[None, ..., None], [max_shape, max_shape], "bilinear")) for t in
                        imgs]
                imgs = np.array(imgs)

                assert len(imgs) == len(np.unique(dists)), ("Number of images and number of distances do not match up!"
                                                            "%i, %i") % (len(imgs), len(np.unique(dists)))
                print(this_f, "x, y, z: ", x, y, z)

                # Calculate and collect the FOV according to the equations from Accorsi:
                accorsi_fov = [get_accorsi_fov(z, p.b, p.t, p.hm / 2) for z in dists]

                # Apply ISL algorithm and collect results:
                if p.method.upper() == "ISL":
                    # Take care of the different initial guess options:
                    if p.IG_type.lower() == "true":
                        IG_MM = z
                    elif p.IG_type.lower() == "random5":
                        IG_MM = np.random.uniform(-5, 5) + np.array(z)
                    elif p.IG_type.lower() == "random10":
                        IG_MM = np.random.uniform(-10, 10) + np.array(z)
                    elif p.IG_type.lower() == "random15":
                        IG_MM = np.random.uniform(-15, 15) + np.array(z)
                    elif p.IG_type.lower() == "max":
                        ig_temp = np.where(np.transpose(imgs, [1, 2, 0]) == np.nanmax(imgs))
                        IG_PX = np.reshape(ig_temp, (3, -1))[:, 0]
                        IG_MM = dists[IG_PX[2]]

                    pos, r2 = isl(imgs, dists, accorsi_fov, SOURCE_FWHM_MM, z_guess_mm=IG_MM, tofit=p.fit,
                                  debug=p.debug, return_r2=True)
                # Apply center-of-mass localization method:
                elif p.method.upper() == "COM":
                    debug_fn = "./%s" % this_f[:-4]
                    pos, r2 = com_vtk(imgs, dists, accorsi_fov, debug=p.debug, debug_fn=debug_fn), np.NAN

                # Store the results for later analysis:
                est_pos.append(pos)
                r2s.append(r2)
                gts.append([x, y, z])

    # Assemble dataframe for the results and analyze the errors:
    est_pos = np.array(est_pos)
    gts = np.array(gts)

    l2_errs = np.sqrt(np.sum((est_pos - gts) ** 2, axis=1))
    error_x, error_y, error_z = np.hsplit((est_pos - gts), 3)

    # Calculate error contributions to the Euclidean distance between GT and estimation in percent:
    cont_eucl = (gts - est_pos) ** 2 / l2_errs.reshape(-1, 1) ** 2 * 100.0
    cont_x, cont_y, cont_z = np.hsplit(cont_eucl, 3)

    # percent_error for x, y and z:
    err_l1_percent_x, err_l1_percent_y, err_l1_percent_z = (np.abs(error_x[:, 0] / gts[:, 0])*100, np.abs(error_y[:, 0] / gts[:, 1])*100,
                                                            np.abs(error_z[:, 0] / gts[:, 2])*100)

    resf = pd.DataFrame({"GTX": gts[:, 0], "GTY": gts[:, 1], "GTZ": gts[:, 2],
                         "EstPosX": est_pos[:, 0], "EstPosY": est_pos[:, 1], "EstPosZ": est_pos[:, 2],
                         "Fit": p.fit, "R2": r2s, "IG": p.IG_type,
                         "ErrorX": error_x.squeeze(), "ErrorY": error_y.squeeze(), "ErrorZ": error_z.squeeze(),
                         "ErrorL1PercentX": err_l1_percent_x.squeeze(), "ErrorL1PercentY": err_l1_percent_y.squeeze(),
                         "ErrorL1PercentZ": err_l1_percent_z.squeeze(),
                         "EuclideanDist": l2_errs,
                         "ErrContL2PercentX": cont_x.squeeze(), "ErrContL2PercentY": cont_y.squeeze(), "ErrContL2PercentZ": cont_z.squeeze(),
                         "ROId_mm": p.ROId_mm,
                         "Method": p.method.upper()})

    # Only inlucde the given 18 sources that are fully within the FOV:
    resfex = resf[(resf['GTY'] == 0) & (resf['GTZ'] == 20)]
    resfex = resfex._append(resf[(resf['GTY'] == 0) & (resf['GTZ'] == 50)])
    resfex = resfex._append(resf[(resf['GTY'] == 2) & (resf['GTZ'] == 50)])
    resfex = resfex._append(resf[(resf['GTY'] == 4) & (resf['GTZ'] == 50)])
    resfex = resfex._append(resf[(resf['GTY'] == 6) & (resf['GTZ'] == 50)])
    resfex = resfex._append(resf[(resf['GTY'] == 8) & (resf['GTZ'] == 50)])
    resfex = resfex._append(resf[(resf['GTY'] == 0) & (resf['GTZ'] == 75)])
    resfex = resfex._append(resf[(resf['GTY'] == 2) & (resf['GTZ'] == 75)])
    resfex = resfex._append(resf[(resf['GTY'] == 4) & (resf['GTZ'] == 75)])
    resfex = resfex._append(resf[(resf['GTY'] == 6) & (resf['GTZ'] == 75)])
    resfex = resfex._append(resf[(resf['GTY'] == 8) & (resf['GTZ'] == 75)])
    resfex = resfex._append(resf[(resf['GTY'] == 0) & (resf['GTZ'] == 100)])
    resfex = resfex._append(resf[(resf['GTY'] == 2) & (resf['GTZ'] == 100)])
    resfex = resfex._append(resf[(resf['GTY'] == 4) & (resf['GTZ'] == 100)])
    resfex = resfex._append(resf[(resf['GTY'] == 6) & (resf['GTZ'] == 100)])
    resfex = resfex._append(resf[(resf['GTY'] == 8) & (resf['GTZ'] == 100)])
    resfex = resfex._append(resf[(resf['GTY'] == 14) & (resf['GTZ'] == 100)])
    resfex.reset_index()

    # Store resulting dataframe in the clipboard:
    resfex.round(2).to_clipboard()

    print("Resulting table: ")
    print(resfex)

    print(
        f"Euclidean distance: average=({np.nanmean(resfex['EuclideanDist']):.2f} ± {np.nanstd(resfex['EuclideanDist']):.2f})mm. "
        f"Median: {np.nanmedian(resfex['EuclideanDist']):.2f}mm. Range: [{np.nanmin(resfex['EuclideanDist']):.2f}, "
        f"{np.nanmax(resfex['EuclideanDist']):.2f}]mm. n={np.sum(~np.isnan(resfex['EuclideanDist']))}. Number of NANs: "
        f"{np.sum(np.isnan(resfex['EuclideanDist']))}. Total: {len(resfex['EuclideanDist'])}\n"
        f"X-component: ({np.mean(resfex['ErrorL1PercentX']):.2f} ± {np.std(resfex['ErrorL1PercentX']):.2f})%, Median: {np.median(resfex['ErrorL1PercentX']):.2f}%\n"         
        f"Y-component: ({np.mean(resfex['ErrorL1PercentY']):.2f} ± {np.std(resfex['ErrorL1PercentY']):.2f})%, Median: {np.median(resfex['ErrorL1PercentY']):.2f}%\n"         
        f"Z-component: ({np.mean(resfex['ErrorL1PercentZ']):.2f} ± {np.std(resfex['ErrorL1PercentZ']):.2f})%, Median: {np.median(resfex['ErrorL1PercentZ']):.2f}%")
    return resfex



