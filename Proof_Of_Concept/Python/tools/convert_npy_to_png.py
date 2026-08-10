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

    input_data = np.load(args.input, allow_pickle=False)

    validate_input_data(input_data)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    plt.imsave(
        args.output,
        input_data,
        cmap=args.cmap
    )

    rows, columns = input_data.shape

    print(f"Wrote {rows}x{columns} image to {args.output}")


###############################################################################
# Function Name: parse_arguments
# Description:
###############################################################################

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Convert a .npy input image into a viewable .png image."
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Path to the .npy image."
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
        help="Matplotlib colormap (e.g. hot, gray, viridis). Default gray."
    )

    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Input image file does not exist: {args.input}")

    if args.output is None:
        args.output = args.input.with_suffix(".png")

    return args


###############################################################################
# Function Name: validate_input_data
# Description:
###############################################################################

def validate_input_data(input_data):
    if input_data.ndim != 2:
        raise ValueError(
            f"Input data must be two-dimensional; received shape {input_data.shape}."
        )

    if not np.all(np.isfinite(input_data)):
        raise ValueError("Input data contains NaN or infinite values.")

    if np.any(input_data < 0.0):
        raise ValueError("Input data contains negative values.")


if __name__ == "__main__":
    main()