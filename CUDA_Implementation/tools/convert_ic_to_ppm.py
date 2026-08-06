###############################################################################
# File Name: convert_ic_to_ppm.py
# Description:
# Converts an arbitrary image (any size or format) into a single-channel
# "P5" PPM file that libwb (and the CUDA implementation) imports as the
# target IC layout.
###############################################################################

###############################################################################
# IMPORTS
###############################################################################

# Library Imports
import argparse
from pathlib import Path

from PIL import Image, ImageOps

# File Imports
import constants


###############################################################################
# GLOBAL FUNCTIONS
###############################################################################

def main():
    args = parse_arguments()

    ic_image = load_grayscale_image(
        args.input,
        args.width,
        args.height,
        args.invert,
        args.binarize,
        args.threshold
    )

    write_p5_ppm(args.output, ic_image)

    print(f"Wrote {ic_image.width}x{ic_image.height} P5 PPM to {args.output}")


###############################################################################
# Function Name: parse_arguments
# Description:
###############################################################################

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Convert an image into a 1-channel P5 .ppm IC layout."
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Path to the source image (.png, .jpg, .bmp, ...)."
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output .ppm path. Defaults to CUDA input_data/<name>.ppm."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Resize width in pixels. Must be used with --height."
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Resize height in pixels. Must be used with --width."
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="Resize to a square of this size (overrides --width/--height)."
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert black and white (flips IC layout polarity)."
    )
    parser.add_argument(
        "--binarize",
        action="store_true",
        help="Threshold to pure black/white (0 or 255)."
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=128,
        help="Threshold [0-255] used when --binarize is set (default 128)."
    )

    args = parser.parse_args()

    # A square --size is shorthand for equal width and height.
    if args.size is not None:
        if args.size <= 0:
            parser.error("--size must be greater than zero.")

        args.width = args.size
        args.height = args.size

    # Width and height only make sense together.
    if (args.width is None) != (args.height is None):
        parser.error("--width and --height must be used together (or use --size).")

    if args.width is not None and (args.width <= 0 or args.height <= 0):
        parser.error("--width and --height must be greater than zero.")

    if args.threshold < 0 or args.threshold > 255:
        parser.error("--threshold must be between 0 and 255.")

    if not args.input.is_file():
        parser.error(f"Input image does not exist: {args.input}")

    # Default output mirrors the source name inside the CUDA input_data folder.
    if args.output is None:
        constants.CUDA_INPUT_DATA_FOLDER_PATH.mkdir(parents=True, exist_ok=True)
        args.output = constants.CUDA_INPUT_DATA_FOLDER_PATH / (args.input.stem + ".ppm")

    return args


###############################################################################
# Function Name: load_grayscale_image
# Description:
###############################################################################

def load_grayscale_image(input_path, width, height, invert, binarize, threshold):
    # Force a single grayscale channel.
    with Image.open(input_path) as input_image:
        image = input_image.convert("L")

    # Optional resize to the requested dimensions.
    if width is not None and height is not None:
        image = image.resize((width, height), Image.Resampling.LANCZOS)

    # Optional polarity flip.
    if invert:
        image = ImageOps.invert(image)

    # Optional hard threshold into a clean binary layout.
    if binarize:
        image = image.point(lambda pixel: 255 if pixel >= threshold else 0)

    return image


###############################################################################
# Function Name: write_p5_ppm
# Description:
###############################################################################

def write_p5_ppm(output_path, image):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width, height = image.size

    # libwb expects: "P5", "<width> <height>", "<max value>", then raw bytes.
    header = f"P5\n{width} {height}\n255\n".encode("ascii")

    # "L" mode packs one byte per pixel in row-major order.
    pixels = image.tobytes()

    with open(output_path, "wb") as file:
        file.write(header)
        file.write(pixels)


if __name__ == "__main__":
    main()