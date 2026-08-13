###############################################################################
# File Name: generate_PSF_data.py
# Description: 
###############################################################################

###############################################################################
# IMPORTS
###############################################################################

# Library Imports
import math
import matplotlib.pyplot as plt
import numpy as np

# File Imports
import constants


###############################################################################
# GLOBAL VARIABLES
###############################################################################


# Generating Double Gausian PSF Data From Equation
# Equation:
#         Normalized value  * (forward scatter range  +  backward scatter range)
# f(r) = (1 / (pi * (1+n))) * (((1/(a^2) * e^(-r^2/a^2)) + ((n/(b^2) * e^(-r^2/b^2)))


###############################################################################
# GLOBAL FUNCTIONS
###############################################################################

def main():
    # Verifying output data has a place to be stored
    constants.DATA_OUTPUT_FOLDER_PATH.mkdir(parents=True, exist_ok=True)

    # Generates 1,000 distances from the beam center, ranging from
    # 0.001 um to 100 um, with the exponents evenly spaced.
    r_values = np.logspace(-3, 2, 1000) # Radial distance from point of exposure

    psf_values = populate_psf_list(r_values)

    psf_mask = generate_psf_mask(constants.MASK_SIZE, constants.PIXEL_SIZE_UM)

    np.save(constants.PSF_2D_OUTPUT_DATA_PATH, psf_mask)

    show_plots(r_values, psf_values, psf_mask)


###############################################################################
# Function Name: populate_psf_list
# Description: 
###############################################################################

def populate_psf_list(r_values):
    psf_vals = []
    
    for r in r_values:
        psf_val = calculate_double_gaussian(r)
        psf_vals.append(psf_val)
    
    return psf_vals


###############################################################################
# Function Name: calculate_double_gaussian
# Description: 
###############################################################################

def calculate_double_gaussian(r):
    normalized_value = 1 / (math.pi * (1 + constants.N))

    forward_scatter = (1 / constants.ALPHA**2) * math.exp(-(r**2) / constants.ALPHA**2)

    backward_scatter = (constants.N / constants.BETA**2) * math.exp(-(r**2) / constants.BETA**2)

    double_gaussian = normalized_value * (forward_scatter + backward_scatter)

    return double_gaussian


###############################################################################
# Function Name: generate_psf_mask
# Description: 
###############################################################################

def generate_psf_mask(mask_size, pixel_size):
    # Creates a 2D mask with all 0's
    psf_mask = np.zeros((mask_size, mask_size))

    # Finds the center of the mask
    center = int(mask_size / 2)

    for row in range(mask_size):
        for column in range(mask_size):
            # Calculate the horizontal and vertical physical distances
            # from this mask location to the beam center.
            # Uses pixel_size as a conversion from pixels to real distanc
            x_distance = (column - center) * pixel_size
            y_distance = (row - center) * pixel_size

            # Calculate the total radial distance from the beam center.
            # True distance from center (in x and y)
            r = math.sqrt(x_distance**2 + y_distance**2)

            # Use the radial distance to calculate this mask value.
            psf_mask[row, column] = calculate_double_gaussian(r)

    return psf_mask


###############################################################################
# Function Name: show_plots
# Description: 
###############################################################################

def show_plots(r_values, psf_values, psf_mask):
    # Create both plots before displaying either one.
    show_1D_psf_values(r_values, psf_values)
    show_2D_psf_mask(psf_mask)

    # Display all currently created figure windows at once.
    plt.show()


###############################################################################
# Function Name: show_1D_psf_values
# Description: 
###############################################################################

def show_1D_psf_values(r_values, psf_values):
    # Create a separate window for the 1D plot.
    plt.figure()

    plt.loglog(r_values, psf_values)

    plt.title("Double-Gaussian PSF")
    plt.xlabel("Radius r (um)")
    plt.ylabel("Energy density")
    plt.grid(True)

    plt.savefig(
        constants.PSF_1D_OUTPUT_IMAGE_PATH,
        dpi=300,
        bbox_inches="tight"
    )


###############################################################################
# Function Name: show_2D_psf_mask
# Description: 
###############################################################################

def show_2D_psf_mask(psf_mask):
    # Create a separate window for the 2D plot.
    plt.figure()

    log_mask = np.log10(psf_mask)

    plt.imshow(log_mask, cmap="hot")
    plt.colorbar(label="log10(Energy density)")

    plt.title("2D Double-Gaussian PSF Mask")
    plt.xlabel("Column")
    plt.ylabel("Row")

    plt.savefig(
        constants.PSF_2D_OUTPUT_IMAGE_PATH,
        dpi=300,
        bbox_inches="tight"
    )


if __name__ == "__main__":
    main()