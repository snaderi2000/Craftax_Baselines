"""Cross-check the C feature port and concrete C/JAX Classic transitions."""
import os
os.environ['JAX_PLATFORMS']='cpu';os.environ['CUDA_VISIBLE_DEVICES']=''
import ctypes as ct,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import jax,jax.numpy as jnp
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax_classic.renderer import render_craftax_symbolic
from craftax.craftax_classic.constants import Achievement
from symbolic_map_memory import ClassicMapMemoryWrapper,MapMemoryState
from symbolic_survival_map import ClassicSurvivalMapWrapper,SurvivalMapState
from wrappers import ClassicLavaPenaltyWrapper
OUT=ROOT/'concept_mapping/runs/pufferlib_map_validation'
lib=ct.CDLL(str(OUT/'bridge.so'));ptr=lambda a:a.ctypes.data_as(ct.c_void_p)
lib.bridge_set.argtypes=[ct.c_void_p]*5;lib.bridge_get.argtypes=[ct.c_void_p]*6
lib.features_step.argtypes=[ct.c_void_p,ct.c_int,ct.c_int,ct.c_int,ct.c_int]
base=make_craftax_env_from_name('Craftax-Classic-Symbolic-v1',auto_reset=False)
mm=ClassicMapMemoryWrapper(base,33);sv=ClassicSurvivalMapWrapper(mm)
@jax.jit
def ref_feature(local,p,action,m,s,reset):
 m=m.replace(position=p,last_position=p,elapsed=jnp.where(reset,0,m.elapsed+1))
 augmented,m=mm.observe(local,m)
 dp=p-s.env_state.position;blocked=(action>=1)&(action<=4)&jnp.all(dp==0)&~s.last_sleeping&~reset
 d=jnp.where(local[1336:1339]<=0,jnp.minimum(s.depleted_steps+1,256),0)
 streak=jnp.where(blocked,jnp.minimum(s.blocked_streak+1,64),0)
 ss=SurvivalMapState(m,d,local[1344]>0,streak)
 full=sv.augment(augmented,ss,action,dp,blocked)
 # Reset action/displacement slots are zero, exactly as initialize().
 full=full.at[-21:-2].set(jnp.where(reset,jnp.zeros(19),full[-21:-2]))
 return full,m,ss
rng=np.random.default_rng(712);world=rng.integers(0,17,(64,64));world[24:42,8:56]=2;world[32,10]=3
feature_errors=[];frames=0
for mode in [0,1]:
 lib.features_clear();p=np.array([32,10],np.int32)
 m=MapMemoryState(None,jnp.zeros((64,64),jnp.uint8),jnp.full((64,64),-1,jnp.int32),jnp.array(p),jnp.array(p),jnp.int32(0))
 s=SurvivalMapState(m,jnp.zeros(3,jnp.int32),jnp.bool_(False),jnp.int32(0))
 for t in range(160):
  a=int(rng.integers(0,17));p=(np.array([32,min(10+t,55)],np.int32) if t<50 else np.clip(p+rng.integers(-1,2,2),0,63).astype(np.int32))
  local=np.zeros(1345,np.float32)
  for i in range(63):
   rr,cc=p+[i//9-3,i%9-4];tile=world[rr,cc] if 0<=rr<64 and 0<=cc<64 else 1;local[i*21+tile]=1
  local[1335:1339]=rng.integers(0,3,4)/10;local[1339+int(rng.integers(0,4))]=1;local[1344]=(t%15<3)
  c=np.zeros(3563,np.float32);c[:1345]=local;lib.features_step(ptr(c),int(p[0]),int(p[1]),a,mode)
  expected,m,s=ref_feature(jnp.asarray(local),jnp.asarray(p),a,m,s,t==0)
  expected=np.asarray(expected).copy()
  if not mode:expected[-40:]=0
  err=float(np.max(np.abs(c-expected)));feature_errors.append(err);frames+=1
  assert np.allclose(c,expected,atol=2e-6,rtol=1e-6),(mode,t,err,np.where(np.abs(c-expected)>2e-6))
feature_result={'passed':True,'frames':frames,'max_absolute_error':max(feature_errors),'modes':['masked','enabled'],'covers':['observed terrain','33x33 crop','age','global water routes including water outside crop','depletion','previous action','displacement','sleep and blocked moves','reset inputs']}
(OUT/'feature_parity.json').write_text(json.dumps(feature_result,indent=2)+'\n')
print('Feature parity:',feature_result,flush=True)
# Load identical supplied states into both engines, avoiding RNG-dependent mobs.
params=base.default_params.replace(spawn_cow_chance=0.,spawn_zombie_base_chance=0.,spawn_zombie_night_chance=0.,spawn_skeleton_chance=0.)
_,template=base.reset(jax.random.PRNGKey(50),params)
ints=['player_health','player_food','player_drink','player_energy'];floats=['player_recover','player_hunger','player_thirst','player_fatigue','light_level']
inv_names=list(template.inventory.__dataclass_fields__)
step=jax.jit(ClassicLavaPenaltyWrapper(base,2).step)
render=jax.jit(render_craftax_symbolic)
def clean():
 changes={k:getattr(template,k).replace(mask=jnp.zeros_like(getattr(template,k).mask)) for k in ['zombies','cows','skeletons','arrows']}
 grid=np.full((64,64),4,np.int32);grid[32,32]=2
 return template.replace(map=jnp.asarray(grid),mob_map=jnp.zeros((64,64),bool),player_position=jnp.array([32,32]),player_direction=jnp.int32(2),player_health=jnp.int32(9),player_food=jnp.int32(9),player_drink=jnp.int32(9),player_energy=jnp.int32(9),is_sleeping=jnp.bool_(False),player_recover=jnp.float32(0),player_hunger=jnp.float32(0),player_thirst=jnp.float32(0),player_fatigue=jnp.float32(0),light_level=jnp.float32(1),inventory=template.inventory.replace(**{k:jnp.int32(0) for k in inv_names}),achievements=jnp.zeros(22,bool),growing_plants_mask=jnp.zeros(10,bool),timestep=jnp.int32(0),**changes)
def load(s):
 x=np.array([*np.asarray(s.player_position),int(s.player_direction),*[int(getattr(s,k)) for k in ints],int(s.is_sleeping),int(s.timestep)],np.int32)
 f=np.array([float(getattr(s,k)) for k in floats],np.float32);inv=np.array([int(getattr(s.inventory,k)) for k in inv_names],np.int32);ach=np.asarray(s.achievements,np.int32);grid=np.asarray(s.map,np.uint8)
 lib.bridge_set(ptr(grid),ptr(x),ptr(f),ptr(inv),ptr(ach))
def get():
 x=np.empty(9,np.int32);f=np.empty(7,np.float32);inv=np.empty(12,np.int32);ach=np.empty(22,np.int32);obs=np.empty(3563,np.float32);grid=np.empty((64,64),np.uint8)
 lib.bridge_get(*[ptr(a) for a in [x,f,inv,ach,obs,grid]])
 return x,f,inv,ach,obs,grid
cases=[]
def add(name,action=0,target=None,inventory=None,**fields):
 s=clean().replace(**{k:jnp.asarray(v,dtype=getattr(clean(),k).dtype) for k,v in fields.items()})
 if target is not None:s=s.replace(map=s.map.at[32,33].set(target))
 if inventory:s=s.replace(inventory=s.inventory.replace(**{k:jnp.int32(v) for k,v in inventory.items()}))
 cases.append((name,s,action))
add('normal_noop');add('blocked_move',2);add('move_grass',2,2);add('lava_death_minus2',2,14)
add('food_floor',player_food=0,player_hunger=25.);add('water_floor',player_drink=0,player_thirst=20.)
add('energy_floor',player_energy=0,player_fatigue=30.);add('sleep_from_positive_fatigue',6,player_energy=3,player_fatigue=20.)
add('sleep_recovery',0,is_sleeping=True,player_energy=3,player_fatigue=-10.)
add('wake_up',0,is_sleeping=True,player_energy=9)
add('deprivation_death',player_health=1,player_food=0,player_recover=-15.)
add('timeout',timestep=9999)
add('drink',5,3,player_drink=2);add('eat_ripe_plant',5,16,player_food=1)
add('mine_stone',5,4,inventory={'wood_pickaxe':1})
add('mine_coal',5,8,inventory={'wood_pickaxe':1})
add('mine_iron',5,9,inventory={'stone_pickaxe':1})
add('mine_diamond',5,10,inventory={'iron_pickaxe':1})
add('place_table',8,2,inventory={'wood':2})
add('make_wood_pick',11,11,inventory={'wood':1})
add('diamond_repeat',5,10,inventory={'iron_pickaxe':1})
name,s,a=cases[-1];cases[-1]=(name,s.replace(achievements=s.achievements.at[19].set(True)),a)
for label,sleep in [('zombie_attack',False),('sleeping_zombie_attack',True)]:
 s=clean();mob=s.zombies.replace(position=s.zombies.position.at[0].set(jnp.array([32,33])),health=s.zombies.health.at[0].set(5),mask=s.zombies.mask.at[0].set(True),attack_cooldown=s.zombies.attack_cooldown.at[0].set(0))
 cases.append((label,s.replace(zombies=mob,mob_map=s.mob_map.at[32,33].set(True),is_sleeping=jnp.bool_(sleep),player_energy=jnp.int32(3)),0))
s=clean();s=s.replace(map=s.map.at[32,33].set(15),growing_plants_positions=s.growing_plants_positions.at[0].set(jnp.array([32,33])),growing_plants_age=s.growing_plants_age.at[0].set(599),growing_plants_mask=s.growing_plants_mask.at[0].set(True))
cases.append(('plant_matures',s,0))
results=[]
for name,s,action in cases:
 load(s)
 if 'zombie_attack' in name:lib.bridge_mob(0,32,33,5,0)
 if name=='plant_matures':lib.bridge_plant(32,33,599)
 before=get()[4][:1345];obs_error=float(np.max(np.abs(before-np.asarray(render(s)))))
 lib.bridge_step(action);x,f,inv,ach,obs,grid=get()
 _,new,reward,done,_=step(jax.random.PRNGKey(6),s,jnp.int32(action),params)
 expected_x=np.array([*np.asarray(new.player_position),int(new.player_direction),*[int(getattr(new,k)) for k in ints],int(new.is_sleeping),int(new.timestep)])
 fields=['row','col','direction',*ints,'is_sleeping','timestep'];diff={fields[i]:{'c':int(x[i]),'jax':int(expected_x[i])} for i in np.flatnonzero(x!=expected_x)}
 for i,k in enumerate(floats):
  if not np.isclose(f[i],float(getattr(new,k)),atol=1e-5):diff[k]={'c':float(f[i]),'jax':float(getattr(new,k))}
 if not np.array_equal(grid,np.asarray(new.map)):diff['terrain']='different'
 if not np.array_equal(inv,[int(getattr(new.inventory,k)) for k in inv_names]):diff['inventory']={'c':inv.tolist(),'jax':[int(getattr(new.inventory,k)) for k in inv_names]}
 if not np.array_equal(ach,np.asarray(new.achievements)):diff['achievements']={'c':np.flatnonzero(ach).tolist(),'jax':np.flatnonzero(new.achievements).tolist()}
 if not np.isclose(f[5],float(reward),atol=1e-5):diff['reward']={'c':float(f[5]),'jax':float(reward)}
 if bool(f[6])!=bool(done):diff['done']={'c':bool(f[6]),'jax':bool(done)}
 results.append({'case':name,'transition_matches':not diff,'differences':diff,'initial_observation_max_error':obs_error,'c_reward':float(f[5]),'jax_reward':float(reward),'terminal':bool(done)})
 print(name,diff,flush=True)
report={'feature_port':feature_result,'transition_cases':results,'all_tested_transitions_match':all(r['transition_matches'] for r in results),'observation_precision_note':'C scalars are float32; JAX normalizes inventory/vitals in float16 then promotes, causing errors up to about 0.0002.','scope':'Identical supplied states, including adjacent zombie attacks and plant growth; not full stochastic parity. World generators and RNG differ.'}
(OUT/'parity_report.json').write_text(json.dumps(report,indent=2)+'\n')
