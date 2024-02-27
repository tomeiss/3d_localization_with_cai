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

from vtkmodules.vtkIOXML import vtkXMLStructuredGridWriter, vtkXMLUnstructuredGridWriter
from vtkmodules.vtkCommonDataModel import vtkStructuredGrid
from vtkmodules.vtkFiltersCore import vtkThreshold, vtkConnectivityFilter, vtkStaticCleanUnstructuredGrid
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.util import numpy_support
import numpy as np
# FIX FOR THE "`np.bool` was a deprecated alias for the builtin `bool`" warning:
np.bool = bool


def getAccorsiGrid(dims, fovs, dists):
    """ dims should be (x, y, z), e.g. (256, 256, 18) """
    nx, ny, nz = dims
    XX, YY, ZZ = np.zeros((3, nx, ny, nz), np.float64)
    for l in range(0, nz):
        fov = fovs[l]
        x, y, z = np.meshgrid(np.linspace(-fov / 2, fov / 2, nx), np.linspace(-fov / 2, fov / 2, nx),
                              dists[l])
        XX[:, :, l] = x[:, :, 0]
        YY[:, :, l] = y[:, :, 0]
        ZZ[:, :, l] = z[:, :, 0]
    return XX, YY, ZZ

def tmSave(f="./vtk_testing_grid.vts", vts_obj=None):
    writer = vtkXMLStructuredGridWriter()
    writer.SetInputData(vts_obj)
    writer.SetFileName(f)
    writer.Write()


def tmSaveUnstr(f, vts_obj):
    writer = vtkXMLUnstructuredGridWriter()
    writer.SetInputData(vts_obj)
    writer.SetFileName(f)
    writer.Write()


def tmExtractPointsAndData(vtk_obj):
    # Get point coordinates and its data, when there are any points in the struct:
    if vtk_obj.GetPoints() is not None:
        pts = numpy_support.vtk_to_numpy(vtk_obj.GetPoints().GetData())
        data = numpy_support.vtk_to_numpy(vtk_obj.GetPointData().GetScalars())
    else:
        pts = np.array((np.NAN, np.NAN, np.NAN)).reshape(-1, 3)
        data = np.array(np.NAN)

    # Convert to our default precision type np.float32:
    pts = pts.astype(np.float32)
    data = data.astype(np.float32)
    return pts, data


def tmCurviStructuredGrid(XX, YY, ZZ, pointData: dict):
    assert XX.shape == YY.shape == ZZ.shape

    # Generate empty structured grid:
    grid = vtkStructuredGrid()
    grid.SetDimensions(XX.shape)

    # Convert xyz-coordinates to a large list of points:
    x_vals = np.vstack([XX.ravel(), YY.ravel(), ZZ.ravel()]).transpose()

    # Sort ravel points first by y-, then by x- and finally by z-coordinate
    sorted_ind = np.lexsort((x_vals[:, 1], x_vals[:, 0], x_vals[:, 2]))
    x_vals = x_vals[sorted_ind]

    # Create a vtkPoints object and write them into the vtk object:
    points = vtkPoints()
    for p in x_vals:
        points.InsertNextPoint(p)
    grid.SetPoints(points)

    # Set point data:
    for this_key in pointData:
        data = pointData[this_key].astype(float)

        # Flatten data:
        data = data.ravel()

        # Apply same order to the pointData:
        data = data[sorted_ind]

        # Convert to vtkFloatArray:
        data = numpy_support.numpy_to_vtk(data)
        data.SetName(this_key)

        # Attach data to the vtk object
        grid.GetPointData().SetScalars(data)

    return grid


def tmThreshold(vtk_obj, scalar_name, lb=None, ub=None):
    # vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS corresponds to a 0:
    FIELD_ASSOCIATION_POINTS = 0
    threshold = vtkThreshold()
    threshold.SetInputData(vtk_obj)

    if lb is not None:
        threshold.SetLowerThreshold(lb)
    if ub is not None:
        threshold.SetUpperThreshold(ub)

    threshold.SetInputArrayToProcess(0, 0, 0, FIELD_ASSOCIATION_POINTS, scalar_name)
    threshold.Update()

    # Get the output vtkUnstructuredGrid after thresholding
    return threshold.GetOutput()


def tmLargestConnectedRegion(vtk_obj):
    connectivity = vtkConnectivityFilter()
    connectivity.SetExtractionModeToLargestRegion()
    connectivity.SetInputData(vtk_obj)
    connectivity.Update()
    return connectivity.GetOutput()

def tmCleanUnstructuredGrid(vtk_unstr):
    """ Removes points of the given unstructured grid that are not used by any cell """
    cleaner = vtkStaticCleanUnstructuredGrid()
    cleaner.SetInputData(vtk_unstr)
    cleaner.Update()
    return cleaner.GetOutput()

if __name__ == '__main__':

    numpy_array = np.load("imgs.npy")
    XX, YY, ZZ = np.load("./XX_YY_ZZ.npy")

    grid = tmCurviStructuredGrid(XX, YY, ZZ, {"reconstruction": numpy_array})
    # tmSave("./input.vts", grid)

    # Get the output tmUnstructuredGrid after thresholding
    output_grid = tmThreshold(grid, "reconstruction", lb=20_000, ub=100_000)
    # tmSaveUnstr("./output1.vtu", output_grid)

    output_grid = tmLargestConnectedRegion(output_grid)
    # tmSaveUnstr("./output2.vtu", output_grid)

    pts, data = tmExtractPointsAndData(output_grid)

    print(pts, data)


