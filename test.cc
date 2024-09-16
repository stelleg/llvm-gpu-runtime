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

  int n = 1<<20; 
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
  std::cout << "Checking results...";
  for(int i=0; i<n; i++){
    if(z[i] != x[i] + y[i]){
      std::cout << "failure: "; 
      printf("%d: %f != %f\n",i, z[i],x[i] + y[i]); 
      exit(1); 
    }
  }
  std::cout << "success" << std::endl; 
}


