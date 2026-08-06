###############################################################################
# File Name: convert_psf_to_raw.py
# Description:
# Generates the double-Gaussian PSF mask and writes it as a libwb ".raw"
# text file that the CUDA implementation imports as the convolution mask.
###############################################################################

###############################################################################
# IMPORTS
###############################################################################

# Library Imports
import argparse
from pathlib import Path

import numpy as np

# File Imports
import constants
from generate_PSF_data import generate_psf_mask


###############################################################################
# GLOBAL FUNCTIONS
###############################################################################

def main():
    args = parse_arguments()

    # Either reuse a previously generated mask or build one from the parameters.
    if args.npy is not None:
        psf_mask = np.load(args.npy)
    else:
        psf_mask = generate_psf_mask(args.mask_size, args.pixel_size)

    # The CUDA convolution kernel does not normalize the mask, so it is
    # normalized here to keep deposited energy on the same 0-to-1 scale as the
    # IC layout (matching the Python proof-of-concept).
    if not args.no_normalize:
        psf_mask = psf_mask / np.sum(psf_mask)

    write_libwb_raw(args.output, psf_mask)

    rows, columns = psf_mask.shape
    print(f"Wrote {rows}x{columns} PSF mask to {args.output}")


###############################################################################
# Function Name: parse_arguments
# Description:
###############################################################################

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate a double-Gaussian PSF mask as a libwb .raw file."
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output .raw path. Defaults to CUDA input_data/PSF_Mask_<n>x<n>.raw."
    )
    parser.add_argument(
        "--mask-size",
        type=int,
        default=constants.MASK_SIZE,
        help=f"PSF mask width/height in pixels (default {constants.MASK_SIZE})."
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=constants.PIXEL_SIZE_UM,
        help="Physical size of one pixel in micrometers."
    )
    parser.add_argument(
        "--npy",
        type=Path,
        default=None,
        help="Load the mask from an existing .npy instead of regenerating it."
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Do not normalize the mask so its values sum to 1."
    )

    args = parser.parse_args()

    # Default output name encodes the mask dimensions for clarity.
    if args.output is None:
        constants.CUDA_INPUT_DATA_FOLDER_PATH.mkdir(parents=True, exist_ok=True)
        args.output = (
            constants.CUDA_INPUT_DATA_FOLDER_PATH
            / f"PSF_Mask_{args.mask_size}x{args.mask_size}.raw"
        )

    return args


###############################################################################
# Function Name: write_libwb_raw
# Description:
###############################################################################

def write_libwb_raw(output_path, matrix):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows, columns = matrix.shape

    with open(output_path, "w") as file:
        # First line holds the dimensions, matching libwb's raw format.
        file.write(f"{rows} {columns}\n")

        # Remaining lines hold the row-major values as ASCII floats.
        for row in range(rows):
            values = " ".join(
                f"{matrix[row, column]:.9g}" for column in range(columns)
            )
            file.write(values + "\n")


if __name__ == "__main__":
    main()
