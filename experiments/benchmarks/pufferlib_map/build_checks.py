"""Build the CPU environment bridge and independent CUDA encoder check."""
from pathlib import Path
import json,subprocess
ROOT=Path(__file__).resolve().parents[3];repo=ROOT/'third_party/PufferLib';out=ROOT/'concept_mapping/runs/pufferlib_map_validation';here=Path(__file__).resolve().parent
for autoreset in [False,True]:
 cmd=['gcc','-shared','-fPIC','-O2',*[f'-I{p}' for p in [repo,repo/'src',repo/'vendor',repo/'raylib-5.5_linux_amd64/include']],str(here/'bridge.c'),str(repo/'raylib-5.5_linux_amd64/lib/libraylib.a'),'-lGL','-lm','-lpthread','-ldl','-o',str(out/('autoreset_bridge.so' if autoreset else 'bridge.so'))]
 if autoreset:cmd+=['-DKEEP_AUTORESET']
 subprocess.run(cmd,check=True)
cmd=json.loads((out/'build_command.json').read_text())
cmd=[str(here/'check_encoder.cu') if x.endswith('/map_validation/pufferl.cu') else x for x in cmd if x!='-DPUFFERLIB_BUILD_MAIN']
cmd[cmd.index('-o')+1]='check_map_encoder'
subprocess.run(cmd,cwd=repo,check=True)
