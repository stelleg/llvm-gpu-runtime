#include<iostream>
#include<dlfcn.h>
#include<llvm/IR/LegacyPassManager.h>
#include<llvm/IR/Constants.h>
#include<llvm/IR/Instruction.h>
#include<llvm/IR/Instructions.h>
#include<llvm/IR/Intrinsics.h>
#include<llvm/IR/IntrinsicsAMDGPU.h>
#include<llvm/IR/IRBuilder.h>
#include<llvm/IR/User.h>
#include<llvm/Transforms/Utils/BasicBlockUtils.h>
#include<llvm/IR/Verifier.h>
#include<llvm/Option/ArgList.h>
#include<llvm/Support/Program.h>
#include<llvm/Support/MemoryBuffer.h>
#include<llvm/Support/TargetSelect.h>
#include<llvm/Support/CommandLine.h>
#include<llvm/Support/raw_os_ostream.h>
#include<llvm/Target/TargetMachine.h>
#include<llvm/Support/ToolOutputFile.h>
#include<llvm/ADT/StringExtras.h>
#include<llvm/MC/TargetRegistry.h>
#include<clang/Basic/TargetID.h>

/*
void llvm-gpu-debug(const char* msg){
  if(const char* env_p = std::getenv("DEBUG_LLVM_HIP")){
    std::cout << msg << std::endl;
  }

}
*/
//#include<llvm/Transforms/IPO/PassManagerBuilder.h>
#define __HIP_PLATFORM_AMD__ 1
#include<hip/hip_runtime_api.h>

#define declare(name) decltype(name)* name##_p = NULL
#define tryLoad(name) name##_p = (decltype(name)*)dlsym(hiphandle, #name)


void* hiphandle; 
declare(hipGetDevice);
declare(hipGetDeviceCount);
declare(hipGetDevicePropertiesR0600); // fucking rocm
declare(hipStreamCreate);
declare(hipModuleLoadData);
declare(hipModuleLaunchKernel);
declare(hipModuleGetFunction);
declare(hipGetErrorString);
declare(hipStreamSynchronize);
declare(hipStreamDestroy);
declare(hipInit);
hipError_t (*hipHostMalloc_p)(void** res, size_t n, int f);


void checkHIP(hipError_t in){
  if(in !=  HIP_SUCCESS){
    std::cerr << "Error: " << hipGetErrorString_p(in) << std::endl;
    exit(in);
  }
}

using namespace llvm; 

int initHIP(){ 
	if(hiphandle) return true;
  hiphandle = dlopen("libamdhip64.so", RTLD_LAZY); 
	if(!hiphandle) return false; 
	tryLoad(hipGetDevice);
	tryLoad(hipGetDeviceCount);
	tryLoad(hipGetDevicePropertiesR0600);
	tryLoad(hipStreamCreate);
	tryLoad(hipStreamDestroy);
	tryLoad(hipStreamSynchronize);
	tryLoad(hipModuleLoadData);
	tryLoad(hipModuleLaunchKernel);
	tryLoad(hipModuleGetFunction);
	tryLoad(hipInit); 
	tryLoad(hipGetErrorString);
  hipHostMalloc_p = (decltype(hipHostMalloc_p))(dlsym(hiphandle, "hipHostMalloc")); 
	checkHIP(hipInit_p(0)); 
	int count;
	checkHIP(hipGetDeviceCount_p(&count)); 
	if(count == 0) return false; 
  return hiphandle != NULL;
}

void* hipManagedMalloc(size_t n){
	void* res;
	checkHIP(hipHostMalloc_p(&res, n, 0));
	return res;
}

void* launchHIPKernel(llvm::Module& m, void** args, size_t n) {
  LLVMContext& ctx = m.getContext(); 
  legacy::PassManager PM;
  legacy::FunctionPassManager FPM(&m);
	int deviceId; 		
	checkHIP(hipGetDevice_p(&deviceId)); 
	hipDeviceProp_t prop;
	checkHIP(hipGetDevicePropertiesR0600_p(&prop, deviceId));
	std::string gcnarch = prop.gcnArchName; 
  StringMap<bool> featureMap; 
  Triple TT("amdgcn", "amd", "amdhsa"); 
  std::optional<StringRef> targetId = clang::parseTargetID(TT, gcnarch, &featureMap); 

  if(!targetId) {
    std::cerr << "Failed to parse target gpu arch" << std::endl; 
    exit(1);
  }

  StringRef cpu = *targetId; 
  std::string featureStr = "";
  for(auto &p : featureMap){
    if(featureStr != "") featureStr += ","; 
    std::string enabled = p.getValue() ? "+" : "-";  
    featureStr += enabled + p.getKey().str(); 
  }
  StringRef features(featureStr); 

  StringRef gpuarch = *targetId;
    

  std::cout << "gcn arch: " << cpu.str() << std::endl; 
  std::cout << "gcn features: " << features.str() << std::endl; 

  m.setTargetTriple(TT.str()); 
  
  Function& F = *m.getFunction("kitsune_kernel");

  AttrBuilder Attrs(ctx);
  Attrs.addAttribute("target-cpu", cpu);
  Attrs.addAttribute("target-features", features);
  /*
  Attrs.addAttribute(Attribute::NoRecurse); 
  Attrs.addAttribute(Attribute::Convergent); 
  */
  F.removeFnAttr("target-cpu");
  F.removeFnAttr("target-features");
  F.setCallingConv(llvm::CallingConv::AMDGPU_KERNEL); 
  F.addFnAttrs(Attrs);

  auto tid = Intrinsic::getDeclaration(&m, Intrinsic::amdgcn_workitem_id_x);
  auto ntid = Intrinsic::getDeclaration(&m, Intrinsic::amdgcn_workgroup_id_x); 

  IRBuilder<> B(F.getEntryBlock().getFirstNonPHI()); 
  Value *tidv = B.CreateCall(tid, {}); 
  Value *ntidv = B.CreateCall(ntid, {});
  Value *ctaidv = ConstantInt::get(tidv->getType(), 8*prop.warpSize);// B.CreateCall(ctaid, {}); 
  
  Value *tidoff = B.CreateMul(ctaidv, ntidv); 
  Value *gtid = B.CreateAdd(tidoff, tidv); 

  /*
  // accumulate reductions in loop
  //const std::vector<BasicBlock*>& blocks = L->getBlocks(); 
  std::set<std::pair<CallInst*, Type*>> reductions;
  for (BasicBlock &BB : F){
    for (Instruction &I : BB) {
      if(auto ci = dyn_cast<CallInst>(&I)){
        auto f = ci->getCalledFunction(); 
        if(f->getAttributes().hasAttrSomewhere(Attribute::KitsuneReduction)){
          std::cout << "Found reduction var: " << ci->getArgOperand(0)->getName().str() << 
                               "with reduction function: " << f->getName().str() << "\n"; 
          auto ty = ci->getArgOperand(1)->getType(); 
          reductions.insert(std::make_pair(ci, ty)); 
          //TODO: check the type to confirm valid reduction
        }
      }
    }
  }

  // accumulate reductions in epilog loop
  std::cout << "Found " << reductions.size() << " reduction variables in kernel\n"; 

  std::vector<std::tuple<CallInst*, Value* , Value*, Type*>> redMap; 
  for(auto &pair : reductions){
    auto ci = pair.first; 
    auto ptr = ci->getArgOperand(0); 
    auto ty = pair.second; 
    IRBuilder<> BH(NewLoop->getHeader()->getTerminator()); 
    auto lptr = BH.CreateBitCast(
      BH.CreateGEP(ty, al, NewIdx), 
      ptr->getType());                             
    redMap.push_back(std::make_tuple(ci, ptr, al, ty)); 
    // Assume there is more than one element, and
    // use the first element for the first iteration of the loop.
    // roughly: 
    //   red = init; 
    //   forall(i = ...){
    //     red = reduce(red, body(i)); 
    //   }
    //   red = init; 
    //   localred[m+1]; 
    //   
    //   forall(k ∈ 0..m-1){
    //     localred[i] = body(j_0); 
    //     for(j ∈ j_k_1..j_k_l-1)
    //       reduce(localred+i, body(j));
    //   }
    //   for( j ∈ j_k_m .. n )
    //     reduce(localred+m, body(j)); 
    //   }
    //   for(k ∈ 0..m) 
    //     reduce(&red, localred[k]); 
    //
    ptr->replaceUsesWithIf(lptr, [L](Use &u){
      if(auto I = dyn_cast<Instruction>(u.getUser())){
        return L->contains(I->getParent()); 
      } else {
        return false;
      }; 
    });
  }
  */

  // inserts intrinsics
  std::vector<Instruction*> tids; 
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

  for(auto c : tids){
    c->replaceAllUsesWith(gtid); 
    c->eraseFromParent(); 
  }

  if(auto *f = m.getFunction("gtid")) f->eraseFromParent();

  m.print(llvm::errs(), nullptr);

	// Ugh, this sucks. Have to use command line utilities and temporary files
	// despite the code existing in the same repository. Might be worth looking 
	// into how to do this in memory, though I'm not sure about linking.
	std::string ObjectFile = "/tmp/kernel.hip.o";
	std::string LinkedObjectFile = "/tmp/kernel.hip-l.o";
	std::string BundledObjectFile = "/tmp/kernel.hip-b.o"; 
  std::error_code EC;
  sys::fs::OpenFlags OpenFlags = sys::fs::OF_None;
  std::unique_ptr<ToolOutputFile> FDOut =
      std::make_unique<ToolOutputFile>(ObjectFile, EC, OpenFlags);
  raw_pwrite_stream *fostr = &FDOut->os();

  std::string error;
  raw_os_ostream ostr(std::cout); 
  InitializeAllTargets(); 
  InitializeAllTargetMCs(); 
  InitializeAllAsmPrinters(); 

  const Target *Target = TargetRegistry::lookupTarget("", TT, error);
  auto TargetMachine =
      Target->createTargetMachine(TT.getTriple(), cpu,
                                     features, TargetOptions(), Reloc::PIC_,
                                     CodeModel::Small, CodeGenOpt::Aggressive);
  m.setDataLayout(TargetMachine->createDataLayout());

  FPM.doInitialization();
  for (Function &F : m)
    FPM.run(F);
  FPM.doFinalization();
  //PM.add(createVerifierPass());
  bool Fail = TargetMachine->addPassesToEmitFile(
      PM, *fostr, nullptr,
      CodeGenFileType::CGFT_ObjectFile, false);
  assert(!Fail && "Failed to emit AMDGCN");
  // Add function optimization passes.
  PM.run(m);
	FDOut->keep(); 

  std::string clangOffloadBundle  = *sys::findProgramByName("clang-offload-bundler");
	std::string lld = *sys::findProgramByName("ld.lld");

  opt::ArgStringList offloadBundleArgList, lldArgList;	
  std::string cpus = "-plugin-opt=mcpu=" + gcnarch;
	std::string lofs = "-o" + LinkedObjectFile;
	lldArgList.push_back(lld.c_str()); 
	lldArgList.push_back("-shared");	
	lldArgList.push_back(cpus.c_str());	
	lldArgList.push_back("-plugin-opt=-amdgpu-internalize-symbols");	
  lldArgList.push_back("-plugin-opt=O3");	
  lldArgList.push_back("-plugin-opt=-amdgpu-early-inline-all=true");	
	lldArgList.push_back("-plugin-opt=-amdgpu-function-calls=false"); 
	lldArgList.push_back(lofs.c_str()); 
	lldArgList.push_back(ObjectFile.c_str()); 
	lldArgList.push_back(nullptr); 

	auto lldsra = toStringRefArray(lldArgList.data());
  sys::ExecuteAndWait(lld, lldsra);

	// Warning: this changes to from hip- to hipv4- in llvm 13
	std::string targets = "-targets=host-x86_64-unknown-linux,hipv4-" 
		+ m.getTargetTriple() + "--" + gcnarch; 
	std::string input1 = "-input=/dev/null"; 
  std::string input2 = "-input=" + LinkedObjectFile; 
	std::string bundledFileStr = "-output=" + BundledObjectFile; 
	offloadBundleArgList.push_back(clangOffloadBundle.c_str());
	offloadBundleArgList.push_back("-type=o"); 
	offloadBundleArgList.push_back(input1.c_str()); 
	offloadBundleArgList.push_back(input2.c_str()); 
	offloadBundleArgList.push_back(targets.c_str()); 
	offloadBundleArgList.push_back(bundledFileStr.c_str());
	offloadBundleArgList.push_back(nullptr); 

	auto cobsra = toStringRefArray(offloadBundleArgList.data());
	sys::ExecuteAndWait(clangOffloadBundle, cobsra); 
	
  ErrorOr<std::unique_ptr<MemoryBuffer>> BundledBinBuf =
      MemoryBuffer::getFile(BundledObjectFile);
	
	std::string hsaco = BundledBinBuf.get()->getBuffer().str(); 
	
	hipModule_t module;	
	checkHIP(hipModuleLoadData_p(&module, (const void*)hsaco.c_str())); 
	hipFunction_t function; 
	checkHIP(hipModuleGetFunction_p(&function, module, "kitsune_kernel")); 
  hipStream_t stream;
	checkHIP(hipStreamCreate_p(&stream)); 

  int blocksize = 8 * prop.warpSize;  
  printf("running with griddim %ld block %d\n", n/blocksize, blocksize); 
	checkHIP(hipModuleLaunchKernel_p(function, n/blocksize, 1, 1, blocksize, 1, 1, 0, stream, args, NULL)); 

	return (void*) stream; 
}

void waitHIPKernel(void* wait) {
	checkHIP(hipStreamSynchronize_p((hipStream_t)wait));
}
