# Setting up the Project


## Using Virtual Environment
I have found it to be easier to install Python libraries in a Virtual Environment (venv).

### Creating Virtual Environment
Execute the following code in the `../Python` directory
```bash
python3 -m venv .venv
```

### Running Virtual Environment
If you are on Windows, run the following:
```bash
source .\.venv\Scripts\Activate.ps1
```

Linux:
```bash
source .venv/bin/activate
```

### Installing Depenencies
In order to install the Python libraries needed for this project to the venv, execute the following:
```bash
pip install .\requirements.txt  
```


## Generate Data

This is to be done in the `../src` directory.

For generating data, execute:
```bash
python generate_PSF_data.py
```


## Running Algorithms

For obtaining simulation results (with optional flag for showing plots (--show-plots)):
```bash
python simulated_exposure.py <IC_PATH.png> <PSF_PATH.npy> <DWELL_OUTPUT_PATH.npy>
```

An example for the algorithm launch would be:
```bash
python simulated_exposure.py ../input_data/IC128.png ../input_data/PSF_Mask_100kV_1um-HSQ.npy ../output_data/IC128_100kV_1um-HSQ.npy
```


## Converting Output to Readable Format

In order to change the algorithm output from .npy to .png, the `convert_psf_to_png.py` script needs to be used. To launch it, execute the following:

```bash
python convert_psf_to_png.py <DWELL_OUTPUT.npy> -o <DWELL_OUTPUT.png>
```

```bash
python convert_psf_to_png.py ../output_data/IC128_100kV_1um-HSQ.npy -o IC128_100kV_1um-HSQ.png
```


## Running Benchmarking

For executing benchmarking:
```bash
python benchmark.py
```