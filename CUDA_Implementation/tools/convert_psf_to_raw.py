###############################################################################
# File Name: convert_psf_to_raw.py
# Description:
# Converts a double-Gaussian PSF mask stored in a NumPy ".npy" file into a
# libwb ".raw" text file used by the CUDA convolution component.
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


###############################################################################
# GLOBAL FUNCTIONS
###############################################################################

def main():
    args = parse_arguments()

    # Load the previously generated PSF mask.
    psf_mask = np.load(args.npy, allow_pickle=False)

    validate_psf_mask(psf_mask, args.mask_size)

    # The CUDA convolution kernel does not normalize the mask, so it is
    # normalized here to keep deposited energy on the same 0-to-1 scale as the
    # IC layout (matching the Python sequential implementation).
    if not args.no_normalize:
        psf_mask = psf_mask / np.sum(psf_mask, dtype=np.float64)

    # CUDA uses single-precision floating-point values.
    # Ensure all values can be represented by CUDA's single-precision floats.
    if np.any(psf_mask > np.finfo(np.float32).max):
        raise ValueError(
            "PSF mask contains values that cannot be represented as float32."
        )

    # CUDA uses single-precision floating-point values.
    psf_mask = psf_mask.astype(np.float32)

    if not np.all(np.isfinite(psf_mask)):
        raise ValueError(
            "PSF mask contains invalid values after conversion to float32."
        )

    if np.sum(psf_mask, dtype=np.float64) <= 0.0:
        raise ValueError(
            "PSF mask becomes zero after conversion to float32."
        )

    write_libwb_raw(args.output, psf_mask)

    rows, columns = psf_mask.shape
    print(f"Wrote {rows}x{columns} PSF mask to {args.output}")


###############################################################################
# Function Name: parse_arguments
# Description:
###############################################################################

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Convert a double-Gaussian PSF mask into a libwb .raw file."
    )

    parser.add_argument(
        "--npy",
        type=Path,
        required=True,
        help="Path to an existing two-dimensional .npy PSF mask."
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
        help=f"Required PSF width/height in pixels (default {constants.MASK_SIZE})."
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Do not normalize the mask so its values sum to 1."
    )

    args = parser.parse_args()

    if args.mask_size <= 0:
        parser.error("--mask-size must be greater than zero.")

    if not args.npy.is_file():
        parser.error(f"Input PSF file does not exist: {args.npy}")

    # Default output name encodes the mask dimensions for clarity.
    if args.output is None:
        constants.CUDA_INPUT_DATA_FOLDER_PATH.mkdir(parents=True, exist_ok=True)
        args.output = (
            constants.CUDA_INPUT_DATA_FOLDER_PATH
            / f"PSF_Mask_{args.mask_size}x{args.mask_size}.raw"
        )

    return args


###############################################################################
# Function Name: validate_psf_mask
# Description:
###############################################################################

def validate_psf_mask(psf_mask, mask_size):
    if psf_mask.ndim != 2:
        raise ValueError(
            f"PSF mask must be two-dimensional; received shape {psf_mask.shape}."
        )

    expected_shape = (mask_size, mask_size)

    if psf_mask.shape != expected_shape:
        raise ValueError(
            f"PSF mask must be {mask_size}x{mask_size}; "
            f"received shape {psf_mask.shape}."
        )

    if not np.all(np.isfinite(psf_mask)):
        raise ValueError("PSF mask contains NaN or infinite values.")

    if np.any(psf_mask < 0.0):
        raise ValueError("PSF mask contains negative values.")

    psf_sum = np.sum(psf_mask, dtype=np.float64)

    if not np.isfinite(psf_sum):
        raise ValueError("PSF mask sum must be finite.")

    if psf_sum <= 0.0:
        raise ValueError("PSF mask sum must be greater than zero.")


###############################################################################
# Function Name: write_libwb_raw
# Description:
###############################################################################

def write_libwb_raw(output_path, matrix):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows, columns = matrix.shape

    with open(output_path, "w", encoding="utf-8") as file:
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