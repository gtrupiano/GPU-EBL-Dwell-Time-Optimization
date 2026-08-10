###############################################################################
# File Name: convert_npy_to_png.py
# Description:
# Converts a NumPy ".npy" itmes into a viewable .png image.
###############################################################################

###############################################################################
# IMPORTS
###############################################################################

# Library Imports
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


###############################################################################
# GLOBAL FUNCTIONS
###############################################################################

def main():
    args = parse_arguments()

    psf_mask = np.load(args.input, allow_pickle=False)

    validate_psf_mask(psf_mask)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    plt.imsave(
        args.output,
        psf_mask,
        cmap=args.cmap
    )

    rows, columns = psf_mask.shape

    print(f"Wrote {rows}x{columns} PSF image to {args.output}")


###############################################################################
# Function Name: parse_arguments
# Description:
###############################################################################

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Convert a .npy PSF mask into a viewable .png image."
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Path to the .npy PSF mask."
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output .png path. Defaults to <input_name>.png."
    )
    parser.add_argument(
        "--cmap",
        default="gray",
        help="Matplotlib colormap (e.g. hot, gray, viridis). Default hot."
    )

    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Input PSF file does not exist: {args.input}")

    if args.output is None:
        args.output = args.input.with_suffix(".png")

    return args


###############################################################################
# Function Name: validate_psf_mask
# Description:
###############################################################################

def validate_psf_mask(psf_mask):
    if psf_mask.ndim != 2:
        raise ValueError(
            f"PSF mask must be two-dimensional; received shape {psf_mask.shape}."
        )

    if not np.all(np.isfinite(psf_mask)):
        raise ValueError("PSF mask contains NaN or infinite values.")

    if np.any(psf_mask < 0.0):
        raise ValueError("PSF mask contains negative values.")


if __name__ == "__main__":
    main()