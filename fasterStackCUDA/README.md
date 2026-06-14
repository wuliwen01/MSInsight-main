# fasterStackCUDA

## How to compile

Use the following commands to compile the .cu files to .dll files with shared option

nvcc jSSA_ttsm.cu -o jSSA.dll --shared
nvcc SSA.cu -o SSA.dll --shared
nvcc jSSA_ttsm_b.cu -o jSSA.dll --shared

## Files

- jSSA_ttsm.cu: CUDA implementation of jSSA.
- jSSA_ttsm_b.cu: Alternative CUDA implementation of jSSA (see file for details).
- SSA.cu: CUDA implementation of SSA.
