program_NAME := test_gpu
NVCC := nvcc

program_CXX_SRCS := example_xor.cpp
program_CXX_SRCS += $(wildcard src/*.cpp)
program_CXX_INCS := $(wildcard inc/*.h)
program_CXX_OBJS := ${program_CXX_SRCS:.cpp=.o}
program_INCLUDE_DIRS := inc
program_LIBRARY_DIRS := 
program_LIBRARIES := cublas cublas_device curand

NVCCFLAGS := -ccbin g++ -g -G -arch=sm_35 -m64
CPPFLAGS += $(foreach includedir,$(program_INCLUDE_DIRS),-I$(includedir))
LDFLAGS += $(foreach librarydir,$(program_LIBRARY_DIRS),-L$(librarydir))
LDFLAGS += $(foreach library,$(program_LIBRARIES),-l$(library))

.PHONY: all clean

all: $(program_CXX_OBJS)
	$(NVCC) $(NVCCFLAGS) $(program_CXX_OBJS) -o $(program_NAME) $(CPPFLAGS) $(LDFLAGS)

%.o: %.cpp $(program_CXX_INCS)
	$(NVCC) $(NVCCFLAGS) -x cu -dc $< -o $@ $(CPPFLAGS) $(LDFLAGS)

clean:
	@- rm $(program_CXX_OBJS)
	@- rm $(program_NAME)
