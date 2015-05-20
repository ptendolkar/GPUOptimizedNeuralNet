# GPU Optimized Neural Network
This repository contains a C++ library for building artificial neural networks with optional regularizers using the standard backpropogation algorithm.

# Branch Details
* master --> serial version
* gpu_version--> GPU version
* other branches --> for testing purposes only

# Building
Running make in either branch produces the binary on example datasets (either XOR or BREAST_CANCER.)  

# Dependencies
We have built the GPU versions using NVIDIA CUDA 6.5. The gpu_version branch uses the NVIDIA CUDA device parallelism feature that is only supported on devices with compute capabilities 3.5 or higher.

The serial version requires [OPENBLAS](https://github.com/xianyi/OpenBLAS) and [GSL](http://www.gnu.org/software/gsl/) libraries. The makefile contains path variables that must be editted if you have a custom path for these dependencies.

# Collaborators
* Andrew E. Hong
* Prasad A. Tendolkar
