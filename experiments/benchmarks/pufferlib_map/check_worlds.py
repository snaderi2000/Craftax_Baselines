"""Compare terrain composition across 128 independent resets per engine."""
import os
os.environ.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',MPLCONFIGDIR='/tmp/puffer-map-mpl')
import ctypes as ct,json
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp
from craftax.craftax_env import make_craftax_env_from_name
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'concept_mapping/runs/pufferlib_map_validation'
lib=ct.CDLL(str(OUT/'bridge.so'));ptr=lambda a:a.ctypes.data_as(ct.c_void_p)
lib.bridge_set.argtypes=[ct.c_void_p]*5;lib.bridge_get.argtypes=[ct.c_void_p]*6
x=np.array([32,32,4,9,9,9,9,0,0],np.int32);f=np.zeros(7,np.float32);inv=np.zeros(12,np.int32);ach=np.zeros(22,np.int32);obs=np.zeros(3563,np.float32);grid=np.zeros((64,64),np.uint8)
lib.bridge_set(*[ptr(a) for a in [grid,x,f,inv,ach]])
c=[]
for i in range(128):
 lib.bridge_reset();lib.bridge_get(*[ptr(a) for a in [x,f,inv,ach,obs,grid]]);c.append(np.bincount(grid.ravel(),minlength=17))
env=make_craftax_env_from_name('Craftax-Classic-Symbolic-v1',auto_reset=False)
reset_maps=jax.jit(jax.vmap(lambda key:env.reset(key,env.default_params)[1].map))
keys=jax.random.split(jax.random.PRNGKey(91235),128);j=[]
for i in range(0,128,16):
 maps=np.asarray(reset_maps(keys[i:i+16]));j.extend(np.bincount(m.ravel(),minlength=17) for m in maps)
c=np.array(c);j=np.array(j);result={}
for name,tile in [('grass',2),('water',3),('stone',4),('tree',5),('path',7),('coal',8),('iron',9),('diamond',10),('sand',13),('lava',14)]:
 result[name]={'c_mean_tiles':float(c[:,tile].mean()),'jax_mean_tiles':float(j[:,tile].mean()),'c_map_sd':float(c[:,tile].std()),'jax_map_sd':float(j[:,tile].std())}
report={'maps_per_engine':128,'paired_worlds':False,'terrain_counts':result,'source_differences':['C suppresses water/mountains within radius 20 by up to 0.3; JAX uses distance/5 clipped then subtracts up to 1','C enforces water/sand precedence over mountains; JAX mountains overwrite previous tiles','JAX includes both path_x and transposed path_y; C only uses one path noise direction','C adds a diamond only if none exists; JAX always_diamond adds one on a sampled stone cell'],'conclusion':'The C port does not implement the same world generator. Equal numeric seeds do not define paired maps across engines.'}
(OUT/'world_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
