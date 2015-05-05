#!/bin/sh

module load cuda

echo "nvcc -x cu -arch=sm_35 -dc test_main.cpp -o test_main.o -I. -lcudadevrt -lcublas_device -lcurand"
nvcc -x cu -arch=sm_35 -dc test_main.cpp -o test_main.o -I. -lcudadevrt -lcublas_device -lcurand

echo "nvcc -x cu -arch=sm_35 -dc Network.cpp -o Network.o -I. -lcudadevrt -lcublas_device -lcurand"
nvcc -x cu -arch=sm_35 -dc Network.cpp -o Network.o -I. -lcudadevrt -lcublas_device -lcurand

echo "nvcc -x cu -arch=sm_35 -dc Layer.cpp -o Layer.o -I. -lcudadevrt -lcublas_device -lcurand"
nvcc -x cu -arch=sm_35 -dc Layer.cpp -o Layer.o -I. -lcudadevrt -lcublas_device -lcurand

echo "nvcc -x cu -arch=sm_35 -dc DevMatrix.cpp -o DevMatrix.o -I. -lcudadevrt -lcublas_device -lcurand"
nvcc -x cu -arch=sm_35 -dc DevMatrix.cpp -o DevMatrix.o -I. -lcudadevrt -lcublas_device -lcurand

echo "nvcc -x cu -arch=sm_35 -dc Funct.cpp -o Funct.o -I. -lcudadevrt -lcublas_device -lcurand"
nvcc -x cu -arch=sm_35 -dc Funct.cpp -o Funct.o -I. -lcudadevrt -lcublas_device -lcurand

echo "nvcc -x cu -arch=sm_35 -dc Data.cpp -o Data.o -I. -lcudadevrt -lcublas_device -lcurand"
nvcc -x cu -arch=sm_35 -dc Data.cpp -o Data.o -I. -lcudadevrt -lcublas_device -lcurand

echo "nvcc -x cu -arch=sm_35 -dc DevData.cpp -o DevData.o -I. -lcudadevrt -lcublas_device -lcurand"
nvcc -x cu -arch=sm_35 -dc DevData.cpp -o DevData.o -I. -lcudadevrt -lcublas_device -lcurand

echo "nvcc -arch=sm_35 test_main.o Network.o Layer.o DevMatrix.o Funct.o Data.o DevData.o -o test_gpu -lcublas_device -lcurand"
nvcc -arch=sm_35 test_main.o Network.o Layer.o DevMatrix.o Funct.o Data.o DevData.o -o test_gpu -lcublas_device -lcurand
