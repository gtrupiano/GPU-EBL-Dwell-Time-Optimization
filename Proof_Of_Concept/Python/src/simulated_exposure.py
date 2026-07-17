###############################################################################
# File Name: simulated_exposure.py
# Description: 
###############################################################################

###############################################################################
# IMPORTS
###############################################################################

# Library Imports
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.signal import fftconvolve

# File Imports
import constants


###############################################################################
# GLOBAL VARIABLES
###############################################################################


###############################################################################
# GLOBAL FUNCTIONS
###############################################################################

def main():
    # Load the IC layout and convert it to binary values: black = 0, white = 1.
    ic_image = Image.open(constants.IC_IMAGE_PATH).convert("1")
    ic_layout = np.asarray(ic_image, dtype=np.float64)

    # Load the previously generated PSF mask.
    psf_mask = np.load(constants.PSF_2D_OUTPUT_DATA_PATH)

    # Keep deposited energy on approximately the same 0-to-1 scale as the IC.
    psf_mask = psf_mask / np.sum(psf_mask)

    # Convolve the IC exposure layout with the PSF.
    deposited_energy = fftconvolve(
        ic_layout,
        psf_mask,
        mode="same"
    )

    # Error Matrix Calculation
    error_matrix, mse = calculate_error_matrix(
        ic_layout,
        deposited_energy
    )

    print(f"MSE: {mse:.8f}")

    show_plots(
        ic_layout, 
        psf_mask, 
        deposited_energy,
        error_matrix,
        mse
    )


###############################################################################
# Function Name: calculate_error_matrix
# Description: 
###############################################################################

def calculate_error_matrix(ic_layout, deposited_energy):
    error_matrix = np.zeros_like(deposited_energy)
    sum = 0.0

    for row in range(constants.IMAGE_SIZE_PIXELS):
        for col in range(constants.IMAGE_SIZE_PIXELS):
            # Calculate the error matrix by subtracting the deposited energy from the IC layout.
            error_matrix[row][col] = ic_layout[row][col] - deposited_energy[row][col]

            # Getting squared error
            error_val_squared = (error_matrix[row][col])**2
            
            # Calculating Sum
            sum += error_val_squared

    # Calculate the mean squared error of the error matrix.
    mse = sum / (constants.IMAGE_SIZE_PIXELS**2)

    return error_matrix, mse


###############################################################################
# Function Name: show_plots
# Description: 
###############################################################################

def show_plots(ic_layout, psf_mask, deposited_energy, error_matrix, mse):
    # Display the input and result.
    plt.figure()
    plt.imshow(ic_layout, cmap="gray")
    plt.title("IC Layout")
    plt.colorbar()

    plt.figure()
    log_mask = np.log10(psf_mask)
    plt.imshow(log_mask, cmap="hot")
    plt.title("PSF Mask")
    plt.colorbar()

    plt.figure()
    plt.imshow(deposited_energy, cmap="hot")
    plt.title("Deposited Energy")
    plt.colorbar()

    plt.figure()
    plt.imshow(error_matrix ** 2, cmap="hot", vmin=0, vmax=1)
    plt.title(f"Squared Error — MSE: {mse:.6f}")
    plt.colorbar(label="Squared exposure error")

    plt.show()


if __name__ == "__main__":
    main()