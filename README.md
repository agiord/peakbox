# The peak-box approach

This repository contains the script developed for the peak-box approach able to detect multiple peak flow events (link to the paper: https://www.mdpi.com/2073-4433/11/1/2), which is compared to the original peak-box approach developed by Zappa et al. 2013. The file `peakbox_v4.py` contains the actual script, while the jupyter-notebook `peakbox_notebook.ipynb` contains its application to one model initialization. 
##
The algorithm was developed with:
- Python 3.6
- SciPy 1.2.0
  - Matplotlib 3.1.0
  - Numpy 1.14.3
- Pandas 0.24.2
- Scikit-learn 0.21.2

## References:
Zappa, M.; F. Fundel and S. Jaun, 2013: A ‘peak-box’approach for supporting
interpretation and verification of operational ensemble peak-flow forecasts, Hy-
drological Processes, 27, (1), 117–131

Giordani, A.; Zappa, M. and Rotach, M.W., 2019: Estimating Ensemble Flood Forecasts’ Uncertainty: A Novel “Peak-Box” Approach for Detecting Multiple Peak-Flow Events. Atmosphere 2020, 11, 2.
