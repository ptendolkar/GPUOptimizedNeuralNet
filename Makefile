CXX = g++
OPENBLAS = /opt/openblas/0.2.3/gnu4/Opteron
TARGET1 = test.o
GSL = /opt/gsl/1.15/gnu4/
all : $(TARGET1)

$(TARGET1) : test_main.cpp
	$(CXX) -o $(TARGET1) test_main.cpp -g -I$(OPENBLAS)/include -I$(GSL)/include -L$(GSL)/lib -L$(OPENBLAS)/lib -Wl,-rpath=$(OPENBLAS)/lib -lopenblas -lgsl -lgslcblas

clean :
	-rm -f *.o
