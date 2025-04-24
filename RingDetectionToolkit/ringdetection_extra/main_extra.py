# Copyright (C) 2025 a.fiorentino4@studenti.unipi.it
# For license terms see LICENSE file.
#
# SPDX-License-Identifier: GPL-2.0-or-later
#
# This program is free software under GPLv2+
# See https://www.gnu.org/licenses/gpl-2.0.html

# ============================ IMPORTS ============================ #
# Standard library imports

"""
Main Entrypoint for ringdetection_extra

This script is just a check to see if the imports are correct
"""

# Local imports
from ringdetection_extra import calculate_radii_in_kamiokande

# -------------------- Constants & Detector Parameters -------------------- #
N_H2O = 1.33           # Refractive index of water
BETA_NU = 1            # v_neutrino / c

# HyperKamiokande dimensions (in meters)
INNER_DIAMETER = 68    # inner diameter of the detector
INNER_HEIGHT = 71      # inner height of the detector
FIDUCIAL_DIAMETER = 64 # fiducial (active) diameter
FIDUCIAL_HEIGHT = 66   # fiducial (active) height

# Total PMTs on the detector
TOTAL_PMTS = 20000

RMAX_SCALE = 1  # Change this if you want to enlarge rmax because of non-orthogonal hits
ALPHA = 10      # Change if you consider the discrete min radius too small


# ======================== HERE IS THE MAIN ========================#

if __name__ == "__main__":
    R_MIN, R_MAX, RADIUS_SCATTER = calculate_radii_in_kamiokande(n_h2o=N_H2O, beta_nu=BETA_NU,
                             inner_diameter=INNER_DIAMETER, inner_height=INNER_HEIGHT,
                             fiducial_diameter=FIDUCIAL_DIAMETER,
                             total_pmts=TOTAL_PMTS, rmax_scale=RMAX_SCALE, alpha=ALPHA,
                             rounding_mode="ceil",verbose=True) #Change "ceil" to "floor" or "round"
