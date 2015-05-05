NVCC = nvcc
DEPS = Network.h Layer.h DevMatrix.h Funct.h Data.h DevData.h cublas_interface.h
TARGET1 = test_gpu.o

objects = test_main.o Network.o Layer.o DevMatrix.o Funct.o Data.o DevData.o

test_gpu: $(objects)
	echo "i'm here"
	nvcc -arch=sm_35 $(objects) -o test_gpu -lcublas_device -lcurand

%.o: %.cpp $(DEPS)
	nvcc -x cu -arch=sm_35 -dc $< -o $@ -I. -lcudadevrt -lcublas_device -lcurand

clean:
	rm -f *.o test_gpu
