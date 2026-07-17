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


## Running Algorithms

This is to be done in the `../src` directory.

For generating data, execute:
```bash
python .\generate_PSF_data.py
```

For obtaining simulation results:
```bash
python .\simulated_exposure.py
```