# MSInsight

## Project Directory Structure

This project is organized into the following main directories and files:

- **3draytracing/**: Contains CUDA-based 3D ray tracing implementation

  - see readme.md for more details

- **conf/**: Configuration files for the project

  - Various configuration files for different algorithms (jssa, ssa)
  - Station location and velocity model files

- **fasterStackCUDA/**: CUDA implementations for stacking algorithms

  - see README for more details

- **model/**: Contains the trained neural network model

  - bfnet_251104a.pt: Pre-trained BFNet model

- **result/**: Output directory for processing results

- **waveform/**: Contains seismic waveform data files

- **Python Scripts**:
  - config.py: Configuration management
  - data.py: Data handling and processing
  - draw.py: Visualization utilities
  - model_bfnet.py: BFNet model implementation
  - stackCU.py: use SSA.dll to stack
  - stackMechCU.py: use jSSA.dll to stack
  - utils.py: Utility functions

## Getting Started

To test the project, please use the `test.ipynb` notebook which contains comprehensive examples and test cases for all the major functionalities.

## Requirements

- Python 3.x
- CUDA-compatible GPU
- Required Python packages (see requirements.txt if available)

## Usage

1. Install the required dependencies
2. Run the test.ipynb notebook to verify the installation and understand the basic usage
3. Modify the configuration files in the `conf/` directory as needed
4. Process your data using the provided Python scripts
