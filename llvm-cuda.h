#include<llvm/Target/TargetMachine.h>

void* cudaManagedMalloc(size_t n);
bool initCUDA();
void* launchCUDAKernel(llvm::Module& m, void** args, size_t n, void** bin); 
void* launchCudaELF(void* elf, void** args, size_t n); 
void waitCUDAKernel(void* wait); 
uint64_t cudaGridSize(); 
