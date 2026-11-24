import json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];repo=ROOT/'third_party/PufferLib';out=ROOT/'concept_mapping/runs/pufferlib_map_validation'
cmd=json.loads((ROOT/'concept_mapping/runs/pufferlib_v100_benchmark/build_command.json').read_text())
cmd=[str(repo/'build/map_validation/pufferl.cu') if x=='src/pufferl.cu' else '-DENV_HEADER="build/map_validation/craftax_map.h"' if x.startswith('-DENV_HEADER=') else x for x in cmd]
cmd[cmd.index('-o')+1]='puffer_map_v100';cmd+=['-DPUFFER_CLASSIC_MAP']
(out/'build_command.json').write_text(json.dumps(cmd,indent=2)+'\n')
subprocess.run(cmd,cwd=repo,check=True)
