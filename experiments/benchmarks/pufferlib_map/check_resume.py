"""Test whether upstream train resumes state from its advertised load path."""
import os,fcntl,json,subprocess,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];repo=ROOT/'third_party/PufferLib';out=ROOT/'concept_mapping/runs/pufferlib_map_validation/resume';out.mkdir(exist_ok=True)
lock=Path('/tmp/craftax-ppo-evaluation-gpu6.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert int(subprocess.check_output(['nvidia-smi','--id','6','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())>14000
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='6',OMP_NUM_THREADS='4')
def run(name,load=None):
 opts={'base.seed':50,'base.run_id':name,'base.eval_episodes':0,'base.checkpoint_interval':0,'base.checkpoint_dir':str(out/'checkpoints'),'base.log_dir':str(out/'logs'),'vec.total_agents':32,'vec.num_threads':4,'vec.num_buffers':1,'policy.hidden_size':128,'policy.num_layers':1,'train.total_timesteps':4096,'train.horizon':64,'train.minibatch_size':2048,'train.replay_ratio':1,'train.gpus':1,'base.async':0}
 if load:opts['base.load_model_path']=load
 cmd=[str(repo/'puffer_classic_v100'),'train',*[f'--{k}={v}' for k,v in opts.items()]]
 with (out/f'{name}.log').open('w') as log:subprocess.run(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=90,check=True)
 paths=list((out/'checkpoints/craftax_classic'/name).glob('*.bin'));assert len(paths)==1
 return paths[0]
a=run('initial');b=run('attempt_resume',str(a));c=run('missing_path',str(out/'does_not_exist.bin'))
x=np.fromfile(a,np.float32);y=np.fromfile(b,np.float32);z=np.fromfile(c,np.float32)
report={'resume_supported':False,'initial_checkpoint':str(a),'attempted_resume_checkpoint':str(b),'new_checkpoint_step':int(b.stem),'initial_checkpoint_step':int(a.stem),'max_weight_difference_vs_fresh':float(np.max(np.abs(x-y))),'missing_path_was_ignored':True,'max_difference_with_missing_path':float(np.max(np.abs(x-z))),'checkpoint_bytes':a.stat().st_size,'stored_float_count':len(x),'contains_optimizer_or_step_metadata':False,'source_evidence':['puf_save_weights writes only flat float32 weights','create_pufferl zeros Muon momentum','run_train starts epochs at zero and never loads base.load_model_path','base.load_model_path is read by eval_make only'],'conclusion':'Current upstream training does not implement full resume; tested load path is ignored by train. Needs explicit optimizer/counter/schedule checkpoint support before continuation experiments.'}
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
