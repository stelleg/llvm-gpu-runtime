#include<stdlib.h>
#include<stdbool.h>
#include<llvm/IR/Module.h>

int initHIP(){ return false; }
void* launchHIPKernel(llvm::Module&, void** args, size_t n) {return NULL;}
void waitHIPKernel(void* wait) {}
