"""Bounded PufferLib Classic throughput test; no claim of algorithm parity."""
import argparse,configparser,fcntl,json,os,subprocess,time,math,re
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
repo=ROOT/'third_party/PufferLib'
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=HERE/'config.json');p.add_argument('--out',type=Path,default=ROOT/'concept_mapping/runs/pufferlib_v100_benchmark');a=p.parse_args()
out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
cfg=json.loads(a.config.read_text());gpu=cfg['gpu'];binary=repo/cfg.get('binary','puffer_classic_v100')
lock=Path(f'/tmp/craftax-ppo-evaluation-gpu{gpu}.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
free=int(subprocess.check_output(['nvidia-smi','--id',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
assert free>=14000,free
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='16',OPENBLAS_NUM_THREADS='1')
metadata={'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'gpu':subprocess.check_output(['nvidia-smi','--id',str(gpu),'--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip(),'config':cfg,'precision':'float32','limitations':['Different network: MinGRU without persistent terrain map','Muon instead of Adam; PPO variant differs','CPU C simulator port with independent RNG; full transition parity not verified','Default achievement/health reward; our extra lava penalty is absent','Native recipe uses fewer PPO updates per sample'],'results':[]}
metadata['limitations']=cfg.get('limitations',metadata['limitations'])
metadata['binary']=str(binary)
(out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
for variant,overrides in cfg['variants'].items():
 for repeat in range(cfg['repeats']):
  name=f'{variant}_r{repeat}'
  opts={'train.gpus':1,'base.gpu_offset':0,'base.seed':50,'train.total_timesteps':cfg['steps'],'base.eval_episodes':0,'base.checkpoint_interval':0,'base.run_id':name,'base.checkpoint_dir':str(out/'checkpoints'),'base.log_dir':str(out/'logs'),'sweep.downsample':20,**overrides}
  command=[str(binary),'train',*[f'--{k}={v}' for k,v in opts.items()]]
  (out/'progress.json').write_text(json.dumps({'state':'running','variant':variant,'repeat':repeat,'time':time.time()},indent=2)+'\n')
  began=time.monotonic()
  with (out/f'{name}.log').open('w') as log:
   process=subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=cfg['timeout_seconds_per_run'])
  result={'variant':variant,'repeat':repeat,'exit_code':process.returncode,'wall_seconds':time.monotonic()-began,'command':command}
  if process.returncode:
   metadata['results'].append(result);(out/'results.json').write_text(json.dumps(metadata,indent=2)+'\n');raise RuntimeError(f'{name} failed: see log')
  ini=configparser.ConfigParser(interpolation=None);ini.optionxform=str
  ini.read(out/'logs/craftax_classic'/f'{name}.ini')
  metrics={k:[float(v) for v in val.split(',')] for k,val in ini['metrics'].items()}
  assert all(math.isfinite(x) for values in metrics.values() for x in values),name
  result['metrics']=metrics
  result['end_to_end_sps']=int(opts['train.total_timesteps'])/result['wall_seconds']
  result['checkpoint_files']=[str(p) for p in (out/'checkpoints/craftax_classic'/name).glob('*.bin')]
  assert result['checkpoint_files'],name
  weights=np.fromfile(result['checkpoint_files'][-1],dtype=np.float32)
  assert len(weights)>0 and np.isfinite(weights).all(),name
  result['parameter_count']=len(weights);result['checkpoint_finite']=True
  metadata['results'].append(result)
  (out/'results.json').write_text(json.dumps(metadata,indent=2)+'\n')
  print(name,'wall_sps',round(result['end_to_end_sps']),'SPS_bins',metrics.get('SPS'),flush=True)
 # Reload the final checkpoint into the real evaluator and run fresh episodes.
 eval_cmd=[str(binary),'eval',result['checkpoint_files'][-1],'--headless',*[f'--{k}={v}' for k,v in opts.items() if k not in ['base.eval_episodes','vec.total_agents','vec.num_buffers','vec.num_threads','base.seed']],'--base.eval_episodes=32','--vec.total_agents=32','--vec.num_buffers=1','--vec.num_threads=4','--base.seed=58150']
 with (out/f'{variant}_reload_eval.log').open('w') as log:
  subprocess.run(eval_cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180,check=True)
 match=re.search(r'CUDA_EVAL .*score=([0-9.eE+-]+) .*games=(\d+) params=(\d+)',(out/f'{variant}_reload_eval.log').read_text())
 assert match and math.isfinite(float(match[1])) and int(match[2])>=32,variant
 metadata.setdefault('reload_evaluations',[]).append({'variant':variant,'requested_episodes':32,'completed_episodes':int(match[2]),'score':float(match[1]),'command':eval_cmd,'passed':True})
 (out/'results.json').write_text(json.dumps(metadata,indent=2)+'\n')
(out/'progress.json').write_text(json.dumps({'state':'complete','time':time.time()},indent=2)+'\n')
