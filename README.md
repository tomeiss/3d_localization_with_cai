[![DOI](https://zenodo.org/badge/764083643.svg)](https://zenodo.org/doi/10.5281/zenodo.11449543)

# 3d_localization_with_cai
This repository contains both the code and the image data for the IOP Physics in Medicine and Biology publication "3D-Localization of Single Point-Like Gamma Sources with a Coded Aperture Camera"

# Installation
The scripts were implemented and executed with Python 3.8.18. Further, the following packages are required:
- NumPy (1.24.4)
- tensorflow (2.10.1)
- SciPy (1.10.1)
- vtk (9.3.0)
- matplotlib (3.7.2),
- Pandas (2.0.3)
- Seaborn (0.12.2)
- openCV (4.6.0)
- bottleneck (1.3.5)
	
Decompress the provided zip files, so that all tiff files are in `./3d_reconstructions/experimental/` and `./3d_reconstructions/mc_simulation/` accordingly.

# Running the Localization Methods
Running the script "analysis_and_plots.py" collects all the 3D reconstructions and performs successively different analysis. Each time a Pandas dataframe is produced and automatically copied into the clipboard. If that causes problems the corresponding line containing "resfex.round(2).to_clipboard()" can be commented out or deleted. In addition to the dataframe for each analysis a grouped boxplot diagramm is generated and displayed. 

The localization methods Iterative Source Localization (ISL) and the Center Of Mass (COM) methods are implemented in the script "localize.py". 
