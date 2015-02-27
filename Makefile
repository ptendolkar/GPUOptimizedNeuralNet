CXX = g++
OPENBLAS = /opt/openblas/0.2.3/gnu4/Opteron
CXXFLAGS = -I$(OPENBLAS)/include -L$(OPENBLAS)/lib -lopenblas -O3
TARGET1 = nnet.o

all : $(TARGET1)

$(TARGET1) : nnet_main.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET1) nnet_main.cpp

clean :
	-rm -f *.o
