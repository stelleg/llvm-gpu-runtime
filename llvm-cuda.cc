#include<stdbool.h>
#include<sstream>
#include<iostream>
#include<dlfcn.h>
#include<llvm/IR/LegacyPassManager.h>
#include<llvm/IR/Constants.h>
#include<llvm/IR/Instruction.h>
#include<llvm/IR/Verifier.h>
#include<llvm/IR/Instructions.h>
#include<llvm/IR/IRBuilder.h>
#include<llvm/IR/Intrinsics.h>
#include<llvm/IR/IntrinsicsNVPTX.h>
#include<llvm/IRReader/IRReader.h>
#include<llvm/Transforms/Utils/BasicBlockUtils.h>
#include<llvm/Support/TargetSelect.h>
#include<llvm/Support/CommandLine.h>
#include<llvm/Support/raw_os_ostream.h>
#include<llvm/Support/raw_os_ostream.h>
#include<llvm/Target/TargetMachine.h>
#include<llvm/Support/ToolOutputFile.h>
#include<llvm/MC/TargetRegistry.h>
#include<llvm/Support/SourceMgr.h>
#include<llvm/Support/Process.h>
#include<llvm/Linker/Linker.h>
#include<llvm/Transforms/IPO/PassManagerBuilder.h>
#include<llvm/Transforms/IPO.h>
#include<nvPTXCompiler.h>
#include<cuda.h>

// TODO: do better than just global versions of these
CUcontext context;
CUdevice device; 
CUstream stream;
int warpsize; 
int numProcs; 

#define declare(name) decltype(name)* name##_p = NULL; 
#define tryLoad(name) name##_p = (decltype(name)*)dlsym(handle, #name)
void* handle = NULL; 
declare(cuInit); 
declare(cuStreamCreate); 
declare(cuStreamDestroy_v2); 
declare(cuStreamSynchronize); 
declare(cuLaunchKernel); 
declare(cuDeviceGet); 
declare(cuGetErrorName); 
declare(cuModuleLoadDataEx); 
declare(cuModuleGetFunction); 
declare(cuModuleUnload); 
declare(cuCtxCreate_v2); 
declare(cuCtxDestroy_v2); 
declare(cuCtxSetCurrent); 
declare(cuMemAllocManaged); 
declare(cuDeviceGetAttribute); 

using namespace llvm; 

#define CUDA_SAFE_CALL(x)                                               \
    do {                                                                \
        CUresult result = x;                                            \
        if (result != CUDA_SUCCESS) {                                   \
            const char *msg;                                            \
            cuGetErrorName_p(result, &msg);                               \
            printf("error: %s failed with error %s\n", #x, msg);        \
            exit(1);                                                    \
        }                                                               \
    } while(0)

#define NVPTXCOMPILER_SAFE_CALL(x)                                       \
    do {                                                                 \
        nvPTXCompileResult result = x;                                   \
        if (result != NVPTXCOMPILE_SUCCESS) {                            \
            printf("error: %s failed with error code %d\n", #x, result); \
            exit(1);                                                     \
        }                                                                \
    } while(0)

void* cudaManagedMalloc(size_t n){
	CUdeviceptr p;
	CUDA_SAFE_CALL(cuMemAllocManaged_p(&p, n, CU_MEM_ATTACH_HOST));
	return (void*)p;
}

bool initCUDA(){
	if(handle) return true; 
	handle = dlopen("libcuda.so", RTLD_LAZY); 
	tryLoad(cuInit); 
	tryLoad(cuStreamCreate); 
	tryLoad(cuStreamDestroy_v2); 
	tryLoad(cuStreamSynchronize); 
	tryLoad(cuLaunchKernel); 
	tryLoad(cuDeviceGet); 
	tryLoad(cuGetErrorName); 
	tryLoad(cuModuleLoadDataEx); 
	tryLoad(cuModuleGetFunction); 
	tryLoad(cuModuleUnload); 
	tryLoad(cuCtxCreate_v2); 
	tryLoad(cuCtxDestroy_v2); 
	tryLoad(cuCtxSetCurrent); 
  tryLoad(cuMemAllocManaged); 
  tryLoad(cuDeviceGetAttribute); 
  if(!handle) return false; 
  if(!cuInit_p) return false; 
  if(cuInit_p(0) != CUDA_SUCCESS) return false;
  CUDA_SAFE_CALL(cuDeviceGet_p(&device, 0));
  CUDA_SAFE_CALL(cuCtxCreate_v2_p(&context, 0, device));

  cuDeviceGetAttribute_p(&numProcs, CU_DEVICE_ATTRIBUTE_MULTI_PROCESSOR_COUNT, device); 
  cuDeviceGetAttribute_p(&warpsize, CU_DEVICE_ATTRIBUTE_WARP_SIZE, device); 

  return true;
}

std::string cudaarch = "sm_70";
std::string cudafeatures = "+ptx64"; 

void* PTXtoELF(const char* ptx){
	nvPTXCompilerHandle compiler = NULL;
  nvPTXCompileResult status;

  size_t elfSize, infoSize, errorSize;
  char *elf, *infoLog, *errorLog;
  unsigned int minorVer, majorVer;

  std::string gpuName = "--gpu-name=" + cudaarch; 
  //std::string gpuFeatures = "--gpu-features=" + cudafeatures; 

  const char* compile_options[] = { gpuName.c_str(), 
                                    "--generate-line-info", 
                                    "--verbose"
                                  };

  NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetVersion(&majorVer, &minorVer));
  printf("Current PTX Compiler API Version : %d.%d\n", majorVer, minorVer);
  printf("ptx being compiled: \n %s", ptx); 
  NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerCreate(&compiler,
                                              (size_t)strlen(ptx),  /* ptxCodeLen */
                                              ptx)                  /* ptxCode */
                          );

  status = nvPTXCompilerCompile(compiler,
                                3,                 /* numCompileOptions */
                                compile_options);  /* compileOptions */

  if (status != NVPTXCOMPILE_SUCCESS) {
    NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetErrorLogSize(compiler, &errorSize));

    if (errorSize != 0) {
      errorLog = (char*)malloc(errorSize+1);
      NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetErrorLog(compiler, errorLog));
      printf("Error log: %s\n", errorLog);
      free(errorLog);
    }
    exit(1);
  }

  NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetCompiledProgramSize(compiler, &elfSize));

  elf = (char*) malloc(elfSize);
  NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetCompiledProgram(compiler, (void*)elf));

  NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetInfoLogSize(compiler, &infoSize));

  if (infoSize != 0) {
    infoLog = (char*)malloc(infoSize+1);
    NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetInfoLog(compiler, infoLog));
    printf("Info log: %s\n", infoLog);
    free(infoLog);
  }

  NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerDestroy(&compiler));

  return elf; 
}

std::string LLVMtoPTX(Module& m) {
  //std::cout << "input module: " << std::endl; 
  //m.print(llvm::errs(), nullptr); 
  LLVMContext& ctx = m.getContext(); 
  int maj, min; 
  cuDeviceGetAttribute_p(&maj, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device); 
  cuDeviceGetAttribute_p(&min, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device); 

  std::ostringstream arch;
  arch << "sm_" << maj << min;
  cudaarch = arch.str();

  Triple TT("nvptx64", "nvidia", "cuda"); 
  m.setTargetTriple(TT.str()); 
  Function& F = *m.getFunction("kitsune_kernel");

  AttrBuilder Attrs(ctx);
  Attrs.addAttribute("target-cpu", cudaarch);
  Attrs.addAttribute("target-features", cudafeatures + ",+" + cudaarch);
  /*
  Attrs.addAttribute(Attribute::NoRecurse); 
  Attrs.addAttribute(Attribute::Convergent); 
  */
  F.removeFnAttr("target-cpu");
  F.removeFnAttr("target-features");
  /*
  F.removeFnAttr(Attribute::StackProtectStrong); 
  F.removeFnAttr(Attribute::UWTable); 
  */
  F.addFnAttrs(Attrs);
  NamedMDNode *Annotations =
    m.getOrInsertNamedMetadata("nvvm.annotations");

  SmallVector<Metadata *, 3> AV;
  AV.push_back(ValueAsMetadata::get(&F));
  AV.push_back(MDString::get(ctx, "kernel"));
  AV.push_back(ValueAsMetadata::get(ConstantInt::get(Type::getInt32Ty(ctx),
                                                     1)));
  Annotations->addOperand(MDNode::get(ctx, AV));
  auto tid = Intrinsic::getDeclaration(&m, Intrinsic::nvvm_read_ptx_sreg_tid_x);
  auto ntid = Intrinsic::getDeclaration(&m, Intrinsic::nvvm_read_ptx_sreg_ntid_x);
  auto ctaid = Intrinsic::getDeclaration(&m, Intrinsic::nvvm_read_ptx_sreg_ctaid_x);

  IRBuilder<> B(F.getEntryBlock().getFirstNonPHI()); 
  Value *tidv = B.CreateCall(tid, {}); 
  Value *ntidv = B.CreateCall(ntid, {});
  Value *ctaidv = B.CreateCall(ctaid, {}); 
  
  Value *tidoff = B.CreateMul(ctaidv, ntidv); 
  Value *gtid = B.CreateAdd(tidoff, tidv); 
  
  // PTXAS doesn't like .<n> global names
  for(GlobalVariable & g : m.globals()){
    auto name = g.getName().str(); 
    for(int i=0; i<name.size(); i++){
      if(name[i] == '.') name[i] = '_'; 
      //std::cout << name << std::endl; 
      g.setName(name); 
    }
  }

  /*
  // accumulate reductions in kernel 
  std::set<CallInst*> reductions;
  for (BasicBlock &BB : F){
    for (Instruction &I : BB) {
      if(auto ci = dyn_cast<CallInst>(&I)){
        auto f = ci->getCalledFunction(); 
        if(f->getAttributes().hasAttrSomewhere(Attribute::KitsuneReduction)){
          std::cout << "Found reduction var: " << ci->getArgOperand(0)->getName().str() << 
                               "with reduction function: " << f->getName().str() << "\n"; 
          reductions.insert(ci); 
          //TODO: check the type to confirm valid reduction
        }
      }
    }
  }

  std::vector<std::tuple<CallInst*, Value* , Value*>> redMap; 
  // for each reduction, insert a tree-based workgroup reduction 
  for(CallInst* ci : reductions){
    // TODO: generic allocation/free calls
    auto ptr = ci->getArgOperand(0); 
    auto op = ci->getArgOperand(1); 
    auto ty = op->getType(); 
    // Create shared memory allocation for each reduction
    
    //auto al = RB.CreateAlloca(ty, nred, ptr->getName() + "_reduction");
    redMap.push_back(std::make_tuple(ci, ptr, al)); 
  }
  */

  // Check if there are unresolved sumbbols to see if we might need libdevice
  std::set<std::string> unresolved; 
  for(auto &f : m) {
    if(f.hasExternalLinkage()){
      unresolved.insert(f.getName().str()); 
    }
  }
   
  if(!unresolved.empty()){
    // Load libdevice and check for provided functions
    llvm::SMDiagnostic SMD; 
    std::optional<std::string> path = sys::Process::FindInEnvPath("CUDA_PATH","nvvm/libdevice/libdevice.10.bc"); 
    if(!path){
      std::cerr << "Failed to find libdevice\n"; 
      exit(1);
    }
    std::unique_ptr<llvm::Module> libdevice =
      parseIRFile(*path, SMD, ctx);
    if(!libdevice){ 
      std::cerr << "Failed to parse libdevice\n"; 
      exit(1);
    }
    // We iterate through the provided functions of the moodule and if there are
    // remaining function calls we add them. 
    std::set<std::string> provided; 
    std::string nvpref = "__nv_"; 
    for(auto &f : *libdevice) {
      std::string name = f.getName().str(); 
      auto res = std::mismatch(nvpref.begin(), nvpref.end(), name.begin()); 
      auto oldName = name.substr(res.second - name.begin()); 
      if(res.first == nvpref.end() && unresolved.count(oldName) > 0) provided.insert(oldName); 
    }

    for(auto &fn : provided){
      if(auto *f = m.getFunction(fn)) f->setName("__nv_" + fn); ; 
    }
    for(auto & F : m){
      for(auto &BB : F){
        for(auto &I : BB){
          if(auto *CI = dyn_cast<CallInst>(&I)){
            if(Function *f = CI->getCalledFunction()){
              //std::cout << f->getName().str() << "\n"; 
            }	
          }
        }
      }
    }
    auto l = Linker(m); 
    l.linkInModule(std::move(libdevice), 2); 
  }

  std::vector<Instruction*> tids; 
  for(auto & F : m){
    for(auto &BB : F){
      for(auto &I : BB){
        if(auto *CI = dyn_cast<CallInst>(&I)){
          if(Function *f = CI->getCalledFunction()){
            if(f->getName() == "gtid"){
              tids.push_back(&I);
            }				
          }	
        }
      }
    }
  }

  for(auto p : tids){
    p->replaceAllUsesWith(gtid); 
    p->eraseFromParent(); 
  }

  if(auto *f = m.getFunction("gtid")) f->eraseFromParent();

  //std::cout << "Module after llvm-gpu processing\n" << std::endl; 
  //m.print(errs(), nullptr);
  //std::cout << std::endl; 
	
  // Create PTX
  auto ptxbuf = new SmallVector<char, 1<<20>(); 
  raw_svector_ostream ptx(*ptxbuf); 

  legacy::PassManager PM;
  legacy::FunctionPassManager FPM(&m); 
  PassManagerBuilder Builder;
  Builder.OptLevel = 2; 
  Builder.VerifyInput = 1; 
  Builder.Inliner = createFunctionInliningPass(Builder.OptLevel); 
  //Builder.populateLTOPassManager(PM);  
  Builder.populateFunctionPassManager(FPM);  
  Builder.populateModulePassManager(PM); 

  // TODO: Hard coded machine configuration, use cuda to check 
  std::string error;
  raw_os_ostream ostr(std::cout); 
  InitializeAllTargets(); 
  InitializeAllTargetMCs(); 
  InitializeAllAsmPrinters(); 

  const Target *PTXTarget = TargetRegistry::lookupTarget("", TT, error);
  if(!PTXTarget){
    std::cerr << error << std:: endl;   
    exit(1);
  }
  auto PTXTargetMachine =
      PTXTarget->createTargetMachine(TT.getTriple(), cudaarch,
                                     "+ptx64", TargetOptions(), Reloc::PIC_,
                                     CodeModel::Small, CodeGenOpt::Aggressive);
  m.setDataLayout(PTXTargetMachine->createDataLayout());

  bool Fail = PTXTargetMachine->addPassesToEmitFile(PM, ptx, nullptr, CodeGenFileType::CGFT_AssemblyFile, false); 
  assert(!Fail && "Failed to emit PTX"); 
  
  FPM.doInitialization();
  for(Function &F : m) FPM.run(F);
  FPM.doFinalization();
  PM.run(m); 
  
  //m.print(llvm::errs(), nullptr); 
  //std::cout << ptx.str().str() << std::endl;
  return ptx.str().str();  
}

CUstream launchCudaELF(void* elf, void** args, size_t n){
  CUmodule module;
  CUfunction kernel;

  CUDA_SAFE_CALL(cuModuleLoadDataEx_p(&module, elf, 0, 0, 0));
  CUDA_SAFE_CALL(cuModuleGetFunction_p(&kernel, module, "kitsune_kernel"));

  CUDA_SAFE_CALL(cuStreamCreate_p(&stream, 0)); 
  // (8 * warpsize) seems like reasonable default block size 
  // TODO: come up with more sophisticated heuristic
  int blocksize = 8 * warpsize;  
  assert(n % blocksize == 0); 
  CUDA_SAFE_CALL(cuLaunchKernel_p(kernel,
                                 n/blocksize, 1, 1, // grid dim
                                 blocksize, 1, 1, // block dim
                                 0, stream, // shared mem and stream
                                 args, NULL)); // arguments

  // Release resources.
  CUDA_SAFE_CALL(cuModuleUnload_p(module));
 
  return stream;
}

void* launchCUDAKernel(Module& m, void** args, size_t n) {
  std::string ptx = LLVMtoPTX(m);
  void* elf = PTXtoELF(ptx.c_str()); 
  return (void*)launchCudaELF(elf, args, n); 
}

void waitCUDAKernel(void* vwait) {
	//CUstream wait = (CUstream)vwait;
	CUstream wait = stream; 
  CUDA_SAFE_CALL(cuStreamSynchronize_p(stream)); 
  //CUDA_SAFE_CALL(cuStreamDestroy_v2_p(wait)); 
  //CUDA_SAFE_CALL(cuCtxDestroy_v2_p(context));
}

uint64_t cudaGridSize(){
  return numProcs * warpSize * 4; 
}

