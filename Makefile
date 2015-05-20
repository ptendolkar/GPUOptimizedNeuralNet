program_NAME := test_serial

# Paths to neccesary libraries
OPENBLAS := /opt/openblas/0.2.3/gnu4/Opteron
GSL := /opt/gsl/1.15/gnu4

# 
program_CXX_SRCS := example_xor.cpp
program_CXX_SRCS += $(wildcard src/*.cpp)
program_CXX_INCS := $(wildcard inc/*.h)
program_CXX_OBJS := ${program_CXX_SRCS:.cpp=.o}
program_INCLUDE_DIRS := $(OPENBLAS)/include $(GSL)/include inc
program_LIBRARY_DIRS := $(OPENBLAS)/lib $(GSL)/lib
program_LIBRARIES := openblas gsl gslcblas rt

CXXFLAGS := -g
CPPFLAGS += $(foreach includedir,$(program_INCLUDE_DIRS),-I$(includedir))
LDFLAGS += $(foreach librarydir,$(program_LIBRARY_DIRS),-L$(librarydir))
LDFLAGS += $(foreach library,$(program_LIBRARIES),-l$(library))

.PHONY: all clean

all: $(program_NAME)

$(program_NAME): $(program_CXX_OBJS)
	$(LINK.cc) $(program_CXX_OBJS) -o $(program_NAME)

%.o: %.cpp $(program_CXX_INCS) 
	$(LINK.cc) -c $< -o $@

clean:
	@- $(RM) $(program_NAME)
	@- $(RM) $(program_CXX_OBJS)
