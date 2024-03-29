#include<llvm/IR/LLVMContext.h>
#include<llvm/IR/Module.h>
#include<llvm/IRReader/IRReader.h>
#include<llvm/Support/SourceMgr.h>
#include "gpu.h"
#include<iostream>

int main(){
	initRuntime(); 	

  llvm::LLVMContext C; 
  llvm::SMDiagnostic SMD; 

  std::unique_ptr<llvm::Module> ExternalModule =
      parseIRFile("kernel.bc", SMD, C);

  int n = 1024; 
  double* x = (double*) gpuManagedMalloc(n*sizeof(double)); 
  double* y = (double*) gpuManagedMalloc(n*sizeof(double)); 
  double* z = (double*) gpuManagedMalloc(n*sizeof(double)); 

  for(int i=0; i<n; i++){
    x[i] = (double)i;
    y[i] = 3.14-(double)i;
  }
  void* args[] = { &x, &y, &z }; 
  auto w = launchKernel(*ExternalModule.get(), args, n); 
  std::cout << "Launched kernel...";
  waitKernel(w); 
  std::cout << "done" << std::endl; 
  for(int i=0; i<n; i++){
    printf("%f\n", z[i]);
  }
}


