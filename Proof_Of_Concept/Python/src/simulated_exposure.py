###############################################################################
# File Name: simulated_exposure.py
# Description:
# Simulates deposited electron-beam energy and iteratively adjusts a
# continuous relative dwell-time map to reduce exposure error.
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

    # Initially, expose the same locations as the desired IC layout.

    # Unlike ic_layout, dwell_time_map will become grayscale
    dwell_time_map = ic_layout.copy()

    print(dwell_time_map.min(), dwell_time_map.max())

    # Iteratively adjust the dwell-time map and record the MSE after each iteration.
    mse_history = optimize_dwell_time(ic_layout, psf_mask, dwell_time_map)

    # Recalculate the final deposited energy after the final dwell-time update.
    deposited_energy = fftconvolve(
        dwell_time_map,
        psf_mask,
        mode="same"
    )

    error_matrix, mse = calculate_error_matrix(
        ic_layout,
        deposited_energy
    )

    print(f"Final MSE: {mse:.8f}")

    show_plots(
        ic_layout,
        dwell_time_map,
        psf_mask,
        deposited_energy,
        error_matrix,
        mse,
        mse_history
    )


###############################################################################
# Function Name: calculate_error_matrix
# Description:
###############################################################################

def calculate_error_matrix(ic_layout, deposited_energy):
    error_matrix = np.zeros_like(deposited_energy)
    error_sum = 0.0

    for row in range(constants.IMAGE_SIZE_PIXELS):
        for col in range(constants.IMAGE_SIZE_PIXELS):
            # Desired exposure minus actual deposited energy.
            error_matrix[row][col] = (ic_layout[row][col] - deposited_energy[row][col])

            # Calculate squared error.
            error_value_squared = error_matrix[row][col] ** 2

            # Add the squared error to the total.
            error_sum += error_value_squared

    # Calculate the mean squared error.
    mse = error_sum / (constants.IMAGE_SIZE_PIXELS ** 2)

    return error_matrix, mse


###############################################################################
# Function Name: optimize_dwell_time
# Description:
###############################################################################

def optimize_dwell_time(ic_layout, psf_mask, dwell_time_map):
    mse_history = []
    best_mse = float('inf') # Python trick to do infinitly large number
    best_dwell_time_map = None


    for iteration in range(constants.MAX_ITERATIONS):
        # Simulate the deposited energy produced by the current dwell-time map (this is what's changing)
        deposited_energy = fftconvolve(
            dwell_time_map,
            psf_mask,
            mode="same"
        )

        # Calculate the error between the desired exposure and actual
        # deposited energy.
        error_matrix, mse = calculate_error_matrix(
            ic_layout,
            deposited_energy
        )

        mse_history.append(mse)

        # Keeping track of the best mse map
        if mse < best_mse:
            best_mse = mse
            best_dwell_time_map = dwell_time_map.copy()

        print(
            f"Iteration {iteration + 1:3d}: "
            f"MSE = {mse:.8f}"
        )

        # Stop early when the MSE becomes sufficiently small.
        if(mse <= constants.MINIMUM_MSE):
            break
        elif iteration > 0 and (abs(mse_history[iteration -1] - mse) < constants.MINIMUM_MSE_CHANGE):
            break
        
        # Convolve the error with the PSF to determine which dwell-time pixels
        # contributed to each exposure error. This accounts for energy spreading
        # into neighboring pixels instead of updating only the same pixel.
        dwell_time_correction = fftconvolve(
            error_matrix,
            psf_mask,
            mode="same"
        )

        # Update the grayscale dwell-time map.

        # Positive error:
        #   Desired energy is greater than deposited energy.
        #   Increase dwell time.

        # Negative error:
        #   Deposited energy is greater than desired energy.
        #   Decrease dwell time.

        # Update each dwell-time pixel using its corresponding error value.
        for row in range(constants.IMAGE_SIZE_PIXELS):
            for col in range(constants.IMAGE_SIZE_PIXELS):
                dwell_time_map[row][col] += (constants.LEARNING_RATE * dwell_time_correction[row][col])

                # Keep the dwell time within its allowed range.
                if dwell_time_map[row][col] < 0.0:
                    dwell_time_map[row][col] = 0.0

                elif dwell_time_map[row][col] > constants.MAX_DWELL_TIME:
                    dwell_time_map[row][col] = constants.MAX_DWELL_TIME

    # Update the dwell map with the best dwell time map to display
    dwell_time_map[:, :] = best_dwell_time_map
    
    print("Min:", dwell_time_map.min())
    print("Max:", dwell_time_map.max())
    print("Middle values:", np.sum(
        (dwell_time_map > 0.0) &
        (dwell_time_map < constants.MAX_DWELL_TIME)
    ))

    print("Zero values:", np.sum(dwell_time_map == 0.0))
    print("Maximum values:", np.sum(
        dwell_time_map == constants.MAX_DWELL_TIME
    ))

    return mse_history


###############################################################################
# Function Name: show_plots
# Description:
###############################################################################

def show_plots(
    ic_layout,
    dwell_time_map,
    psf_mask,
    deposited_energy,
    error_matrix,
    mse,
    mse_history
):
    plt.figure()
    plt.imshow(ic_layout, cmap="gray", vmin=0, vmax=1)
    plt.title("Desired IC Layout")
    plt.colorbar()

    plt.figure()
    plt.imshow(
        dwell_time_map,
        cmap="gray",
        vmin=0,
        vmax=constants.MAX_DWELL_TIME
    )
    plt.title("Optimized Relative Dwell-Time Map")
    plt.colorbar(label="Relative dwell time")

    plt.figure()

    # Prevent log10(0).
    log_mask = np.log10(
        np.maximum(psf_mask, np.finfo(np.float64).tiny)
    )

    plt.imshow(log_mask, cmap="hot")
    plt.title("PSF Mask")
    plt.colorbar()

    plt.figure()
    plt.imshow(deposited_energy, cmap="hot", vmin=0, vmax=1)
    plt.title("Deposited Energy")
    plt.colorbar()

    plt.figure()
    plt.imshow(error_matrix ** 2, cmap="hot", vmin=0, vmax=1)
    plt.title(f"Squared Error — MSE: {mse:.6f}")
    plt.colorbar(label="Squared exposure error")

    plt.figure()
    plt.plot(mse_history)
    plt.title("MSE During Dwell-Time Optimization")
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()