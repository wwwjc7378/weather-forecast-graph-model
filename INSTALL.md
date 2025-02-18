# Installation Guide for STGNNs

## System Requirements

The STGNNs model was developed and tested on the following system:

- **Operating System**: Ubuntu 20.04.2 LTS (Focal Fossa)
- **Architecture**: aarch64 (64-bit ARM)
- **CPU**: 64 cores
- **Memory**: 124 GB total

Note: This system does not have a GPU. The model runs on CPU only.

## Python Version

The code was developed using Python 3.10.14. It is recommended to use this version or later.

To check your Python version, run:

```bash
python --version
```

## Dependencies

The main dependencies for this project are:

- torch
- torch-geometric
- pandas
- numpy
- scikit-learn
- tqdm

To install these dependencies, create a `requirements.txt` file with the following content:

```
torch==1.9.0
torch-geometric==2.0.3
pandas==1.3.3
numpy==1.21.2
scikit-learn==0.24.2
tqdm==4.62.3
```

Then, install the dependencies using:

```bash
pip install -r requirements.txt
```

Note: You may need to install a CPU-only version of PyTorch, as this system does not have a GPU. Please refer to the official PyTorch website for installation instructions specific to your system architecture.

## Installation Time

The installation process typically takes about 10-15 minutes, depending on the system's internet connection speed.

## Running the Demo

After installation, you can run the demo script to test the STGNNs model:

```bash
python demo.py
```

This will load the sample data, prepare the graph, run the model, and display the results.

## Additional Notes

- Ensure you have sufficient disk space for the code, dependencies, and data files.
- If you encounter any issues related to the ARM architecture, you may need to compile some packages from source or find ARM-compatible versions.

If you face any installation issues, please contact the authors for assistance.