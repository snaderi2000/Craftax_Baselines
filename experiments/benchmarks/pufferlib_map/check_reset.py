"""Check episode boundary reset of C map memory, counters and action feedback."""
import ctypes as ct,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'concept_mapping/runs/pufferlib_map_validation'
lib=ct.CDLL(str(OUT/'autoreset_bridge.so'));ptr=lambda a:a.ctypes.data_as(ct.c_void_p)
lib.bridge_set.argtypes=[ct.c_void_p]*5;lib.bridge_get.argtypes=[ct.c_void_p]*6
results=[]
for name,action,tile,timestep in [('timeout',0,4,9999),('lava',2,14,0)]:
 grid=np.full((64,64),4,np.uint8);grid[32,32]=2;grid[32,33]=tile
 x=np.array([32,32,2,9,9,9,9,0,timestep],np.int32);f=np.array([0,0,0,0,1],np.float32);inv=np.zeros(12,np.int32);ach=np.zeros(22,np.int32)
 lib.bridge_set(*[ptr(a) for a in [grid,x,f,inv,ach]])
 lib.bridge_step(action)
 f=np.empty(7,np.float32);obs=np.empty(3563,np.float32)
 lib.bridge_get(*[ptr(a) for a in [x,f,inv,ach,obs,grid]])
 assert f[6]==1 and x[8]==0
 memory=obs[1345:2434].reshape(33,33);outside=memory.copy();outside[13:20,12:21]=0
 assert not np.any(outside)
 assert np.all(obs[-24:]==0) # depletion(3), action(17), movement(2), block(2)
 assert np.all(obs[2434:3523].reshape(33,33)[13:20,12:21]==0)
 results.append({'case':name,'done_preserved':True,'new_episode_timestep':int(x[8]),'memory_cleared':True,'survival_history_cleared':True,'reward':float(f[5])})
(OUT/'reset_report.json').write_text(json.dumps({'passed':True,'cases':results},indent=2)+'\n');print(results)
