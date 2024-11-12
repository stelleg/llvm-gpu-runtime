#include"gpu.h"
#include"llvm-cuda.h"
#include"llvm-hip.h"
#include"llvm-spirv.h"
#include<llvm/IR/Module.h>
#include<llvm/IRReader/IRReader.h>
#include<llvm/Support/SourceMgr.h>
#include<fstream>
#include<error.h>
#include<stdbool.h>
#include<map>

// We keep a map from pointer to bitcode to pointer to elf
std::map<const char*, void*> kernelMap; 

void err(const char* msg){
  return error(1, 1, "%s", msg);
}

typedef enum {
  none,
  spirv,
  hip,
  cuda
} runtime;  

runtime globalRuntime = none;

uint64_t gpuGridSize(){
  switch(globalRuntime){
    case cuda:
      return cudaGridSize();
    case hip:
      return hipGridSize(); 
    default:
      return 1UL<<16; 
  }
}
void *gpuManagedMalloc(uint64_t n){
	switch(globalRuntime){
		case hip:
			return hipManagedMalloc(n);
		case cuda:
			return cudaManagedMalloc(n);
    case spirv:
			err("no spirv managed malloc");
		default:
      return malloc(n); 
	}	
	return NULL;
}

void initRuntime(){
  if(globalRuntime != none) return;
  if(initCUDA()) {
		globalRuntime = cuda;
		return;
	}
  if(initHIP()){
		globalRuntime = hip; 
		return;
	}
  if(initSPIRV()){
		globalRuntime = spirv; 
		return;
	}
	err("No gpu runtimes found, needed OpenCL with SPIRV support, HIP, or CUDA\n");
}

void* launchBCKernel(const char* bc, uint64_t bcsize, void** args, uint64_t n){
  if(auto search = kernelMap.find(bc); search != kernelMap.end()){
    return launchBinKernel(search->second, args, n); 
  }
  llvm::LLVMContext C; 
  llvm::SMDiagnostic SMD; 
  std::string strbuf(bc, bcsize); 
  std::ofstream out("runtime.bc");
  out << strbuf; 
  out.close();

  llvm::StringRef sr(bc, bcsize); 
  llvm::MemoryBufferRef mbr(sr, "kernelModRef"); 
  std::unique_ptr<llvm::Module> mod =
      parseIR(mbr, SMD, C);
  if(!mod){
    SMD.print("Failed to parse kernel IR: ", llvm::errs()); 
    exit(1); 
  }
  void* bin; 
  void* wait = launchKernel(*mod, args, n, &bin); 
  auto p = kernelMap.try_emplace(bc, bin); 
  kernelMap[bc] = bin; 
  return wait; 
}

void* launchBinKernel(void* bin, void** args, uint64_t n){
  switch(globalRuntime){
    case cuda:
      return launchCudaELF(bin, args, n);
    default:
      err("unspported binary launch"); 
  }
  return nullptr; 
}

void* launchKernel(llvm::Module& bc, void** args, uint64_t n, void** bin){
  switch(globalRuntime){
    case spirv: 
      return launchSPIRVKernel(bc, args, n);
    case hip:
      return launchHIPKernel(bc, args, n);
    case cuda:
      return launchCUDAKernel(bc, args, n, bin);
    default:
      err("Can't get kernel without valid runtime");
  }
  return NULL; 
}

void waitKernel(void* wait){
  switch(globalRuntime){
    case spirv:
      return waitSPIRVKernel(wait);
    case hip:
      return waitHIPKernel(wait);
    case cuda:
      return waitCUDAKernel(wait);
    default:
      err("Can't wait kernel without valid runtime");
  }
}
