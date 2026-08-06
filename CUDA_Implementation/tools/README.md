# Setting up the Project

## Using Virtual Environment
I have found it to be easier to install Python libraries in a Virtual Environment (venv).

### Creating Virtual Environment
Execute the following code in the `/tools` directory
```bash
python3 -m venv .venv
```

### Running Virtual Environment
If you are on Windows, run the following:
```bash
.\.venv\Scripts\Activate.ps1
```

Linux:
```bash
source .venv/bin/activate
```

### Installing Depenencies
In order to install the Python libraries needed for this project to the venv, execute the following:
```bash
python -m pip install -r requirements.txt
```


## Running Algorithms

This is to be done in the `../tools` directory.

**NOTE: All original data lives in the Python POC folder due to new Gaussian data being generated with alternate parameters**

Also, examples are done with Linux syntax.

1. For converting IC `.png` to `.ppm`, execute:

    Note: Use `--size`, `--width`, or `--height` only when the source image must be resized.

    ```bash
    python convert_ic_to_ppm.py {file path} --binarize
    ```
    
    Example:
    ```bash
    python convert_ic_to_ppm.py ../../Proof_Of_Concept/Python/input_data/IC128.png --binarize
    ```

    **Explaination:** This will take the IC 128x128 image from the Python implementation (this is where all original data is stored) and convert it to `.ppm` format. Then it will be stored in `input_data` folder to be used for the actual CUDA Implementation.
    <br><br>


2. For converting the PSF mask data `.npy` to `.raw`, execute:
    ```bash
    python convert_psf_to_raw.py --npy {file path}
    ```

    Example:
    ```bash
    python convert_psf_to_raw.py --npy ../../Proof_Of_Concept/Python/output_data/double_gaussian_psf_2D.npy
    ```

    **Explaination:** This loads an existing PSF mask from a NumPy `.npy` file (this is where all original data is stored), verifies that it is 65x65 by default, normalizes the values so their sum is approximately 1, converts them to `float32` for the CUDA implementation, and writes the result to the CUDA `input_data` folder as a `.raw` file.
    <br><br>


3. For converting (PSF mask `.raw`  **OR** dwell output `.raw`) to `.png`, execute:
    
    PSF Mask:
    ```bash
    python convert_raw_to_png.py {file path} --scale auto
    ```

    Dwell Output:
    ```bash
    python convert_raw_to_png.py {file path} --scale dwell
    ```

    Example (PSF Mask):
    ```bash
    python convert_raw_to_png.py ../input_data/PSF_Mask_65x65.raw --scale auto
    ```

    Example (Dwell Output):
    ```bash
    python convert_raw_to_png.py ../output_data/optimized_dwell_time.raw --scale dwell
    ```

    **Explaination:** **Explanation:** This converts a `.raw` matrix into a viewable PNG. PSF masks should use `--scale auto`, while dwell-time outputs should use `--scale dwell`. The PNG defaults to the CUDA `output_data` folder and is only a visualization; the original `.raw` file remains the numerical data.
    <br><br>