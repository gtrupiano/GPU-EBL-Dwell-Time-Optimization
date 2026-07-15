from pathlib import Path

###############################################################################
# CONSTANTS
###############################################################################

# Folder locations
SRC_FOLDER_PATH = Path(__file__).resolve().parent
PYTHON_FOLDER_PATH = SRC_FOLDER_PATH.parent

INPUT_DATA_FOLDER_PATH = PYTHON_FOLDER_PATH / "input_data"
DATA_OUTPUT_FOLDER_PATH = PYTHON_FOLDER_PATH / "output_data"


# Double-Gaussian parameters:
# 100 kV, 50 nm HSQ:

# Range of forward scattering
ALPHA = 0.0024

# Range of backward scattering
BETA = 27.8728

# Ratio of total energy deposited
N = 0.7183

# The 512x512 layout represents a 1 µm x 1 µm area.
IMAGE_WIDTH_UM = 1.0
IMAGE_SIZE_PIXELS = 512

# Physical distance represented by one image pixel.
PIXEL_SIZE_UM = IMAGE_WIDTH_UM / IMAGE_SIZE_PIXELS

# The CUDA paper uses a 65x65 PSF mask.
MASK_SIZE = 65

# Input paths
IC_IMAGE_PATH = INPUT_DATA_FOLDER_PATH / "IC512.png"

# PSF output paths
PSF_1D_OUTPUT_IMAGE_PATH = DATA_OUTPUT_FOLDER_PATH / "double_gaussian_psf_1D.png"
PSF_2D_OUTPUT_DATA_PATH = DATA_OUTPUT_FOLDER_PATH / "double_gaussian_psf_2D.npy"
PSF_2D_OUTPUT_IMAGE_PATH = DATA_OUTPUT_FOLDER_PATH / "double_gaussian_psf_2D.png"

# Exposure simulation output paths
DEPOSITED_ENERGY_OUTPUT_DATA_PATH = DATA_OUTPUT_FOLDER_PATH / "deposited_energy.npy"
DEPOSITED_ENERGY_OUTPUT_IMAGE_PATH = DATA_OUTPUT_FOLDER_PATH / "deposited_energy.png"