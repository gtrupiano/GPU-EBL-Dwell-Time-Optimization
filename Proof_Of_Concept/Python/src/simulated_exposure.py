import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.signal import fftconvolve

import constants


EXPOSURE_SUMMARY_IMAGE_PATH = (
    constants.DATA_OUTPUT_FOLDER_PATH / "exposure_simulation.png"
)


def main():
    constants.DATA_OUTPUT_FOLDER_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    target_pattern = load_binary_image(constants.IC_IMAGE_PATH)
    validate_image_size(target_pattern)

    psf_density = load_psf_mask(constants.PSF_2D_OUTPUT_DATA_PATH)

    # Convert energy density into an approximate discrete pixel weight.
    pixel_area_um2 = constants.PIXEL_SIZE_UM**2
    psf_weights = psf_density * pixel_area_um2

    # Normalize the truncated discrete kernel for this first simulation.
    # This keeps a uniformly exposed area near a value of 1.0.
    psf_kernel = psf_weights / np.sum(psf_weights)

    # The initial dwell-time map is the desired binary pattern.
    exposure_map = target_pattern.copy()

    deposited_energy = fftconvolve(
        exposure_map,
        psf_kernel,
        mode="same"
    )

    # get error matrix
    error_matrix = get_error(
        deposited_energy,
        exposure_map
    )

    # Remove tiny negative FFT roundoff values.
    deposited_energy = np.clip(deposited_energy, 0.0, None)

    np.save(
        constants.DEPOSITED_ENERGY_OUTPUT_DATA_PATH,
        deposited_energy
    )

    print_simulation_information(
        target_pattern,
        psf_kernel,
        deposited_energy
    )

    show_simulation(
        target_pattern,
        psf_kernel,
        deposited_energy,
        error_matrix
    )


###############################################################################
# Function Name: load_binary_image
# Description:
#     Loads the IC image as grayscale and converts black pixels to 0.0 and
#     white pixels to 1.0.
###############################################################################

def load_binary_image(image_path):
    if not image_path.exists():
        raise FileNotFoundError(
            f"IC image was not found:\n{image_path}"
        )

    image = Image.open(image_path).convert("L")
    image_array = np.asarray(image, dtype=np.float64)

    return (image_array >= 128).astype(np.float64)


###############################################################################
# Function Name: load_psf_mask
# Description:
#     Loads the previously generated two-dimensional PSF density mask.
###############################################################################

def load_psf_mask(psf_path):
    if not psf_path.exists():
        raise FileNotFoundError(
            f"PSF file was not found:\n{psf_path}\n"
            "Run generate_PSF_data.py first."
        )

    psf_mask = np.load(psf_path)

    if psf_mask.ndim != 2:
        raise ValueError(
            f"Expected a 2D PSF mask, but loaded shape {psf_mask.shape}."
        )

    if np.any(psf_mask < 0.0):
        raise ValueError("The PSF mask contains negative values.")

    if not np.any(psf_mask > 0.0):
        raise ValueError("The PSF mask contains no positive values.")

    return psf_mask.astype(np.float64)


###############################################################################
# Function Name: validate_image_size
# Description:
#     Confirms that the input image matches the configured IC resolution.
###############################################################################

def validate_image_size(image):
    expected_shape = (
        constants.IMAGE_SIZE_PIXELS,
        constants.IMAGE_SIZE_PIXELS
    )

    if image.shape != expected_shape:
        raise ValueError(
            f"Expected IC image shape {expected_shape}, "
            f"but loaded {image.shape}."
        )


###############################################################################
# Function Name: get_error
# Description:
#     Creates the error matrix, the difference between the
#     deposited_energy and the exposure_map
###############################################################################

def get_error(
    deposited_energy,
    exposure_map
):
    return deposited_energy - exposure_map

###############################################################################
# Function Name: print_simulation_information
# Description:
#     Prints the important simulation dimensions and numerical ranges.
###############################################################################

def print_simulation_information(
    target_pattern,
    psf_kernel,
    deposited_energy
):
    print(f"IC image shape: {target_pattern.shape}")
    print(f"PSF mask shape: {psf_kernel.shape}")

    print(
        "Pixel size: "
        f"{constants.PIXEL_SIZE_UM:.9f} µm "
        f"({constants.PIXEL_SIZE_UM * 1000.0:.6f} nm)"
    )

    print(f"Normalized PSF sum: {psf_kernel.sum():.9f}")

    print(
        "Deposited-energy range: "
        f"{deposited_energy.min():.9f} to "
        f"{deposited_energy.max():.9f}"
    )


###############################################################################
# Function Name: show_simulation
# Description:
#     Displays the desired pattern, PSF mask, and developed-energy image in
#     one figure for direct comparison.
###############################################################################

def show_simulation(
    target_pattern,
    psf_kernel,
    deposited_energy,
    error_matrix
):
    figure, axes = plt.subplots(
        1,
        4,
        figsize=(16, 5),
        constrained_layout=True
    )

    target_plot = axes[0].imshow(
        target_pattern,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest"
    )

    axes[0].set_title("Desired IC Pattern")
    axes[0].set_xlabel("Column")
    axes[0].set_ylabel("Row")
    figure.colorbar(
        target_plot,
        ax=axes[0],
        label="Desired exposure"
    )

    # Use log10 only for displaying the PSF. The convolution still uses the
    # original floating-point PSF values.
    smallest_positive = psf_kernel[psf_kernel > 0.0].min()
    log_psf = np.log10(
        np.maximum(psf_kernel, smallest_positive)
    )

    psf_plot = axes[1].imshow(
        log_psf,
        cmap="plasma",
        interpolation="nearest"
    )

    axes[1].set_title("PSF Filter")
    axes[1].set_xlabel("Column")
    axes[1].set_ylabel("Row")
    figure.colorbar(
        psf_plot,
        ax=axes[1],
        label="log10(PSF weight)"
    )

    developed_plot = axes[2].imshow(
        deposited_energy,
        cmap="plasma",
        vmin=0.0,
        vmax=1.0,
        interpolation="bilinear"
    )

    axes[2].set_title("Developed-Energy Image")
    axes[2].set_xlabel("Column")
    axes[2].set_ylabel("Row")
    figure.colorbar(
        developed_plot,
        ax=axes[2],
        label="Relative deposited energy"
    )

    error_plot = axes[3].imshow(
        error_matrix,
        cmap="plasma",
        vmin=0.0,
        vmax=1.0,
        interpolation="bilinear"
    )

    axes[3].set_title("Error Matrix")
    axes[3].set_xlabel("Column")
    axes[3].set_ylabel("Row")
    figure.colorbar(
        error_plot,
        ax=axes[3],
        label="Error Matrix"
    )

    figure.savefig(
        constants.DEPOSITED_ENERGY_OUTPUT_IMAGE_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


if __name__ == "__main__":
    main()
