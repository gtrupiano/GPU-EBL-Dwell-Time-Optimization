###############################################################################
# File Name: convert_raw_to_png.py
# Description:
# Converts a libwb ".raw" matrix (such as the optimized dwell-time output)
# into a viewable .png image.
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

# File Imports
import constants


###############################################################################
# GLOBAL FUNCTIONS
###############################################################################

def main():
    args = parse_arguments()

    matrix = read_libwb_raw(args.input)

    vmin, vmax = resolve_scale(matrix, args.scale, args.max_dwell)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    plt.imsave(args.output, matrix, cmap=args.cmap, vmin=vmin, vmax=vmax)

    print(f"Wrote {matrix.shape[0]}x{matrix.shape[1]} image to {args.output}")


###############################################################################
# Function Name: parse_arguments
# Description:
###############################################################################

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Convert a libwb .raw matrix into a viewable .png image."
    )

    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=constants.CUDA_OUTPUT_DATA_FOLDER_PATH / "optimized_dwell_time.raw",
        help="Path to the .raw file (default: CUDA optimized_dwell_time.raw)."
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output .png path. Defaults to CUDA output_data/<input_name>.png."
    )
    parser.add_argument(
        "--cmap",
        default="gray",
        help="Matplotlib colormap (e.g. gray, hot, viridis). Default gray."
    )
    parser.add_argument(
        "--scale",
        choices=("auto", "dwell", "unit"),
        default="dwell",
        help="Intensity scaling: auto (min-max), dwell (0..max-dwell), unit (0..1)."
    )
    parser.add_argument(
        "--max-dwell",
        type=float,
        default=constants.MAX_DWELL_TIME,
        help="Upper bound used when --scale dwell is selected."
    )

    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Input raw file does not exist: {args.input}")

    if args.max_dwell <= 0.0:
        parser.error("--max-dwell must be greater than zero.")

    # Defaults output to output_data folder
    if args.output is None:
        constants.CUDA_OUTPUT_DATA_FOLDER_PATH.mkdir(
            parents=True,
            exist_ok=True
        )

        args.output = (
            constants.CUDA_OUTPUT_DATA_FOLDER_PATH /
            f"{args.input.stem}.png"
        )

    return args


###############################################################################
# Function Name: read_libwb_raw
# Description:
###############################################################################

def read_libwb_raw(input_path):
    with open(input_path, "r", encoding="utf-8") as file:
        header = file.readline().split()

        # libwb writes "rows columns", or just "rows" for a single column.
        if len(header) == 1:
            rows, columns = int(header[0]), 1
        elif len(header) == 2:
            rows, columns = int(header[0]), int(header[1])
        else:
            raise ValueError(
                "Raw file header must contain one or two integer dimensions."
            )

        if rows <= 0 or columns <= 0:
            raise ValueError("Raw file dimensions must be greater than zero.")

        values = np.array(file.read().split(), dtype=np.float64)

    expected_values = rows * columns

    if values.size != expected_values:
        raise ValueError(
            f"Expected {expected_values} values for a {rows}x{columns} matrix, "
            f"but found {values.size}."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError("Raw matrix contains NaN or infinite values.")

    return values.reshape(rows, columns)


###############################################################################
# Function Name: resolve_scale
# Description:
###############################################################################

def resolve_scale(matrix, scale, max_dwell):
    if scale == "dwell":
        return 0.0, max_dwell

    if scale == "unit":
        return 0.0, 1.0

    # Auto scaling uses the data range, guarding against a flat image.
    minimum = float(matrix.min())
    maximum = float(matrix.max())

    if maximum <= minimum:
        maximum = minimum + 1.0

    return minimum, maximum


if __name__ == "__main__":
    main()