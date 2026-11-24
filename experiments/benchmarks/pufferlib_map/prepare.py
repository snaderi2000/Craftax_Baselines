"""Generate isolated upstream derivatives with observed-map inputs and lava cost."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'third_party/PufferLib/build/map_validation';OUT.mkdir(parents=True,exist_ok=True)
p=ROOT/'third_party/PufferLib/ocean/craftax_classic/craftax_classic.h'
s=p.read_text()
def change(old,new):
 global s
 assert s.count(old)==1,(old[:80],s.count(old));s=s.replace(old,new)
change('#define OBS_SIZE 1345',f'#include "{HERE}/map_features.h"\n#define OBS_SIZE MAP_FEATURE_OBS\n#ifndef MAP_FEATURE_MODE\n#define MAP_FEATURE_MODE 1\n#endif')
change('    // Scratch for per-step reward computation','    ObservedMap memory;\n    // Scratch for per-step reward computation')
change('    compute_observations(env);\n}', '    map_clear(&env->memory);\n    compute_observations(env);\n    map_augment(&env->memory,env->agents[0].observations,env->player_r,env->player_c,-1,MAP_FEATURE_MODE);\n}')
change('    float r = ach_r + hp_r;', '    float r = ach_r + hp_r;\n    if(in_bounds(env->player_r,env->player_c)&&map_get(env,env->player_r,env->player_c)==BLK_LAVA)r-=2.0f;')
change('        puf_reset(env);   // auto-reset (observation written inside)', '#ifndef PUF_PARITY_NO_RESET\n        puf_reset(env);   // auto-reset (observation written inside)\n#else\n        compute_observations(env);\n        map_augment(&env->memory,env->agents[0].observations,env->player_r,env->player_c,action,MAP_FEATURE_MODE);\n#endif')
change('    } else {\n        compute_observations(env);\n    }\n}', '    } else {\n        compute_observations(env);\n        map_augment(&env->memory,env->agents[0].observations,env->player_r,env->player_c,action,MAP_FEATURE_MODE);\n    }\n}')
(OUT/'craftax_map.h').write_text(s)
(ROOT/'concept_mapping/runs/pufferlib_map_validation/source.json').write_text(json.dumps({'upstream_header_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'generated_header_sha256':hashlib.sha256(s.encode()).hexdigest()},indent=2)+'\n')
print(OUT/'craftax_map.h')
# Keep all upstream files untouched; compile a generated source derivative.
for name in ['pufferl.cu','algo.cu','ocean.cu']:
 text=(ROOT/'third_party/PufferLib/src'/name).read_text()
 if name=='ocean.cu':
  text=text.replace('static void create_custom_encoder(Encoder* enc) {\n#ifdef PUFFER_NETHACK',f'#include "{HERE}/map_encoder.cu"\nstatic void create_custom_encoder(Encoder* enc) {{\n#if defined(PUFFER_CLASSIC_MAP)\n    create_map_encoder(enc);\n#elif defined(PUFFER_NETHACK)')
 (OUT/name).write_text(text)
