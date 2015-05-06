DEPS = Network.h Layer.h DevMatrix.h Funct.h Data.h DevData.h

objects = test_main.o Network.o Layer.o DevMatrix.o Funct.o Data.o DevData.o

all: $(objects)
	nvcc -ccbin g++ -g -G -arch=sm_35 $(objects) -o test_gpu -m64 -lcublas -lcublas_static -lculibos -lcurand

%.o: %.cpp $(DEPS)
	nvcc -ccbin g++ -g -G -x cu -arch=sm_35 -dc $< -o $@ -I. -m64 -lcublas -lcublas_static -lculibos -lcurand 

clean:
	rm -f *.o test_gpu
