"""Build pinned PufferLib Classic in float32 for the server's V100 GPUs."""
import json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
repo=ROOT/'third_party/PufferLib'
cuda=ROOT/'.tools/puffer-cuda'
vidia=Path('/home/shawheen/miniconda3/envs/jax-v100-cuda12/lib/python3.10/site-packages/nvidia')
curand=Path('/home/shawheen/miniconda3/lib/python3.13/site-packages/nvidia/curand')
libs=[vidia/'cublas',vidia/'cusolver',vidia/'nccl',curand]
cmd=[str(cuda/'bin/nvcc'),'-O3','--use_fast_math','-arch=sm_70','--cudart=shared','--cudadevrt=none','-std=c++17','-I.','-Isrc','-Ivendor','-Iraylib-5.5_linux_amd64/include',f'-I{cuda}/include',f'-I{cuda}/include/cccl',*[f'-I{x}/include' for x in libs],'-DENV_HEADER="ocean/craftax_classic/craftax_classic.h"','-DENV_NAME=craftax_classic','-DPUFFER_ENV_NAME="craftax_classic"','-DPUFFER_CRAFTAX_CLASSIC','-DPUFFERLIB_BUILD_MAIN','-DPRECISION_FLOAT','-Xcompiler=-DPLATFORM_DESKTOP','-Xcompiler=-fopenmp','-Xcompiler=-Wno-narrowing','--diag-suppress=2361','--diag-suppress=111','--diag-suppress=128','src/pufferl.cu','raylib-5.5_linux_amd64/lib/libraylib.a',f'-L{cuda}/lib','-lcudart',str(vidia/'cublas/lib/libcublas.so.12'),str(vidia/'cusolver/lib/libcusolver.so.11'),str(vidia/'nccl/lib/libnccl.so.2'),str(curand/'lib/libcurand.so.10'),'-lnvidia-ml','-lGL','-lm','-Xlinker=-lpthread','-Xlinker=-lgomp','-o','puffer_classic_v100']
for p in [cuda/'lib',*[x/'lib' for x in libs],vidia/'cusparse/lib',vidia/'nvjitlink/lib']:
 cmd += ['-Xlinker=-rpath',f'-Xlinker={p}']
cmd=[('-Xlinker='+x) if '.so.' in x and not x.startswith('-') else x for x in cmd]
out=ROOT/'concept_mapping/runs/pufferlib_v100_benchmark'
(out/'build_command.json').write_text(json.dumps(cmd,indent=2)+'\n')
subprocess.run(cmd,cwd=repo,check=True)
