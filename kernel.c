#include"kernelHeader.h"
#include<math.h>

void kitsune_kernel(int id, double* x, double* y, double* z){
	z[id] = y[id] + x[id] + 1; 
}

