# Program Description

## Program Input

- Station coordinates, grid, layer thickness, layer wave velocity

## Program Output

- Theoretical travel time, shape station \* grid point xyz

## Process Flow

### Read Grid Parameters

- The parameters are stored in a txt file, with each line containing a parameter name followed by its value.

### Read Station Coordinates, Determine the Number of z Levels

- The coordinate file has each line formatted as: station ID, x, y, z
- z represents the elevation.
- Divided by 1m precision, with a sampling rate typically under 1000Hz, and the lowest wave velocity around 1000. A 1m/1000mps = 1ms can meet the required precision.

### Calculate Theoretical Travel Time for Each Layer Based on Station z Position

- Input file:
  - `upperbound.txt` contains the elevation of the top surface of the layers.
- Coordinate Conversion:
  - Convert from elevation to layer thickness.
- Traverse station z \* grid layers:
  - Stream parallel computation.
  - For each z and layer, compute horizontal slowness in parallel.
  - Horizontal slowness range:
    - min: 0
    - max: 1/max_vel
- Now calculating the minimum error rho, changed to calculating time and exit angle:
  - Kernel outputs the travel time array of length lrho for the current z and layer.
  - Each kernel's index is the rho array index.
  - blockDim \* threadDim lrho.
- Also need to calculate epicentral distance.

### Traverse Grid Points in Layers, Calculate Epicentral Distance to Each Station, Find the Closest Value, and Calculate Error

- Traverse xy and stations.
- Input: travel_time, exit_angle, epicentral_distance, grid parameters, station parameters.
- Calculate epicentral distance (epdis).
- Find the closest value to epdis in the epicentral_distance array.
- Output corresponding travel_time, exit_angle, and error for the staxy pair.
- Should the result be written directly into the file in the host?

### Final Output

- nsta \* x \* y \* z

compile `RTtraveltime.cpp` to generate `RTtraveltime.dll`, with results returned through pointers for the calling program.

## Compile

To compile the CUDA program and create the necessary executables or shared libraries, follow the steps below.

### 1. **Build CUDA Program**

To compile the CUDA program and generate an executable named `raycuda`, run the following command in your terminal:

```bash
nvcc -arch=sm_75 --shared -o RTtraveltime.dll RTtraveltime.cpp kernel.cu
