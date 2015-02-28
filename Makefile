CXX = g++
OPENBLAS = /opt/openblas/0.2.3/gnu4/Opteron
TARGET1 = nnet.o

all : $(TARGET1)

$(TARGET1) : nnet_main.cpp
	$(CXX) -o $(TARGET1) nnet_main.cpp -O3 -I$(OPENBLAS)/include -L$(OPENBLAS)/lib -Wl,-rpath=$(OPENBLAS)/lib -lopenblas

clean :
	-rm -f *.o
