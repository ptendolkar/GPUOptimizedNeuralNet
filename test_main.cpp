#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include "neuron.h"

int main(int argc, char* argv[])
{
	std::vector<size_t> dimensions(4);
	dimensions[0] = 5;
	dimensions[1] = 3;
	dimensions[2] = 4;
	dimensions[3] = 1; 

	Network testN(dimensions);
	testN.remove(0);

    return 0;
}
