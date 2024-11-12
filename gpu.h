#include<stdint.h>
#include<stddef.h>

#ifdef __cplusplus
namespace llvm {
class Module; 
}
extern "C" {
// if bin isn't passed we don't save the results
void* launchKernel(llvm::Module& bc, void** args, uint64_t n, void** bin = nullptr); 
#endif
void *gpuManagedMalloc(uint64_t n); 
void initRuntime(); 
void* launchBinKernel(void* bc, void** args, uint64_t n); 
void* launchBCKernel(const char* bc, uint64_t bcsize, void** args, uint64_t n); 
void waitKernel(void* wait); 
uint64_t gpuGridSize(); 
#ifdef __cplusplus
}
#endif
