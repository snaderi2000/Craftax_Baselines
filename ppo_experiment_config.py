"""Pure-Python configuration for PPO studies; importing it never initializes JAX."""
from dataclasses import asdict,dataclass,field,fields
from pathlib import Path
import hashlib,json,math,re

DEFAULT_ARCHITECTURES=('mlp','gru','spatial','spatial_gru')
ARCHITECTURES=(*DEFAULT_ARCHITECTURES,'spatial_map_gru','map_mlp','map_mlp_gru','neural_map_gru')
LOG_MILESTONES=(65536,100000,150000,220000,330000,500000,750000,1000000,1500000,
    2200000,3300000,5000000,7500000,10000000,15000000,22000000,33000000,50000000,
    75000000,100000000,150000000,220000000,330000000,500000000,750000000,1000000000,
    1500000000,2200000000,3300000000,5000000000,7500000000)


def construct(cls,value):
    unknown=set(value)-{f.name for f in fields(cls)}
    if unknown:raise ValueError(f'Unknown {cls.__name__} fields: {sorted(unknown)}')
    return cls(**value)


@dataclass(frozen=True)
class Rewards:
    death_penalty:float=0.
    lava_penalty:float=0.
    diamond_reward:float=1.
    plant_reward:float=1.
    all22_terminal_bonus:float|None=None
    map_coverage_bonus:float|None=None

    def validate(self):
        for name,value in asdict(self).items():
            if name in ('all22_terminal_bonus','map_coverage_bonus') and value is None:continue
            if not math.isfinite(value) or value<0:raise ValueError(f'{name} must be a finite nonnegative number')
        if self.death_penalty and self.lava_penalty:raise ValueError('Choose general death cost or lava-only cost')


@dataclass(frozen=True)
class PPO:
    num_envs:int=1024
    num_steps:int=64
    update_epochs:int=4
    num_minibatches:int=8
    gamma:float=.99
    gae_lambda:float=.8
    clip_eps:float=.2
    vf_coef:float=.5
    ent_coef:float=.01
    lr:float=.0002
    lr_horizon_timesteps:int|None=None
    max_grad_norm:float=1.
    layer_size:int=512
    anneal_lr:bool=True
    optimistic_resets:bool=True
    optimistic_reset_ratio:int=16
    map_view_size:int=33
    map_warmstart_crop_from:int|None=None
    map_embedding_size:int=4
    map_global_readout:bool=False
    map_landmarks:str|None=None
    map_single_readout:bool=False
    map_board_only:bool=False
    map_board_attention:bool=False
    map_v1_attention_adapter:bool=False
    map_v1_imitation_init:str|None=None
    map_internal_crop:bool=False
    map_patch_mlp:bool=False
    map_spatial_regions:bool=False
    map_fine_shortcut:bool=False
    map_full_terrain:bool=False
    map_survival_features:str='none'
    neural_map_channels:int=0
    neural_map_reset_every_step:bool=False
    recovery_bank:str|None=None
    recovery_probability:float=0.
    reset_seed_bank:str|None=None
    policy_temperature:float=1.
    reset_gru_every_step:bool=False
    gru_concat_observation:bool=False
    motion_inputs:bool=False
    sru_gru:bool=False
    sru_residual_actor_critic:bool=False
    sru_failure_replay:bool=False
    sru_warmup_steps:int=0
    sru_gate_lr:float=3e-5
    sru_identity_coef:float=0.
    sru_map_coef:float=0.
    sru_lr_anneal_from:int|None=None
    sru_lr_anneal_until:int|None=None
    sru_lr_final_fraction:float=.1
    world_view:str|None=None
    local_policy_init:str|None=None
    ctec_coef:float=0.
    ctec_width:int=1024
    ctec_dim:int=64
    ctec_batch:int=512
    ctec_updates:int=2
    ctec_lr:float=3e-4
    ctec_gamma:float=.3
    ctec_warmup_steps:int=1000000
    ctec_ramp_steps:int=9000000
    ctec_fade_start:float=.8
    ctec_clip:float=5.

    def validate(self):
        if type(self.neural_map_channels) is not int or not 0<=self.neural_map_channels<=64:raise ValueError('neural_map_channels must be 0..64')
        if type(self.sru_failure_replay) is not bool or (self.sru_failure_replay and
                (not self.sru_gru or self.optimistic_resets or self.sru_map_coef)):
            raise ValueError('Failure seed replay requires a standard-reset SRU without a reconstruction loss')
        if not isinstance(self.map_global_readout,bool):raise ValueError('map_global_readout must be boolean')
        if self.map_landmarks is not None and (self.map_landmarks not in ('masked','locations','events') or
                not self.map_global_readout or self.map_single_readout or self.map_full_terrain):
            raise ValueError('Landmarks require a crop plus observed global terrain')
        if not isinstance(self.map_single_readout,bool) or self.map_single_readout and (not self.map_global_readout or self.map_full_terrain):
            raise ValueError('map_single_readout requires observed-only global map input')
        if type(self.map_internal_crop) is not bool or (self.map_internal_crop and (not self.map_single_readout or self.map_patch_mlp)):
            raise ValueError('map_internal_crop requires a single observed map and the original Map-MLP reader')
        if not isinstance(self.map_patch_mlp,bool) or self.map_patch_mlp and not self.map_single_readout:
            raise ValueError('map_patch_mlp requires a single observed map')
        if type(self.map_spatial_regions) is not bool or (self.map_spatial_regions and (
                not self.map_single_readout or self.map_internal_crop or self.map_patch_mlp)):
            raise ValueError('map_spatial_regions requires one observed world map without a crop or patch MLP')
        if type(self.map_fine_shortcut) is not bool or (self.map_fine_shortcut and not self.map_spatial_regions):
            raise ValueError('map_fine_shortcut requires spatial regions')
        if type(self.map_board_only) is not bool or (self.map_board_only and
                (not self.map_single_readout or self.map_internal_crop or self.map_spatial_regions or
                 self.map_patch_mlp or self.map_survival_features != 'none')):
            raise ValueError('map_board_only requires a single observed board without additional readers')
        if type(self.map_board_attention) is not bool or (self.map_board_attention and not self.map_board_only):
            raise ValueError('map_board_attention requires the board-only policy')
        if type(self.map_v1_attention_adapter) is not bool or (self.map_v1_attention_adapter and
                (not self.map_board_only or self.map_board_attention)):
            raise ValueError('Frozen V1 requires its dedicated board-only attention adapter')
        if not isinstance(self.map_full_terrain,bool) or (self.map_full_terrain and not self.map_global_readout):
            raise ValueError('map_full_terrain requires map_global_readout')
        if not isinstance(self.neural_map_reset_every_step,bool):raise ValueError('neural_map_reset_every_step must be boolean')
        if self.map_survival_features not in ('none','masked','enabled'):raise ValueError('Invalid map_survival_features')
        if not isinstance(self.reset_gru_every_step,bool):raise ValueError('reset_gru_every_step must be boolean')
        if not isinstance(self.gru_concat_observation,bool):raise ValueError('gru_concat_observation must be boolean')
        if not isinstance(self.motion_inputs,bool):raise ValueError('motion_inputs must be boolean')
        if type(self.sru_residual_actor_critic) is not bool or (self.sru_residual_actor_critic and not self.sru_gru):
            raise ValueError('Residual SRU heads require an SRU policy')
        if (type(self.sru_gru) is not bool or type(self.sru_warmup_steps) is not int or self.sru_warmup_steps<0
                or not math.isfinite(self.sru_gate_lr) or self.sru_gate_lr<=0
                or not math.isfinite(self.sru_identity_coef) or self.sru_identity_coef<0):
            raise ValueError('Invalid SRU configuration')
        for name in ('num_envs','num_steps','update_epochs','num_minibatches','layer_size','optimistic_reset_ratio','map_embedding_size'):
            value=getattr(self,name)
            if not isinstance(value,int) or isinstance(value,bool) or value<1:raise ValueError(f'{name} must be a positive integer')
        if not math.isfinite(self.sru_map_coef) or self.sru_map_coef < 0 or (self.sru_map_coef and not self.sru_gru):
            raise ValueError('Reconstruction coefficient must be finite, nonnegative, and used with an SRU')
        start,end=self.sru_lr_anneal_from,self.sru_lr_anneal_until
        batch=self.num_envs*self.num_steps
        if start is not None or end is not None:
            if (not self.sru_gru or self.anneal_lr or type(start) is not int or type(end) is not int
                    or not 0<=start<end or start%batch or end%batch
                    or not 0<self.sru_lr_final_fraction<=1):
                raise ValueError('SRU learning-rate schedule requires aligned start/end steps, a positive floor, and fixed base rates')
        if self.num_envs%self.num_minibatches:raise ValueError('Minibatches must contain whole recurrent environment sequences')
        if self.optimistic_resets and self.num_envs%min(self.num_envs,self.optimistic_reset_ratio):
            raise ValueError('Optimistic reset ratio must divide the environment count')
        for name in ('gamma','gae_lambda','clip_eps','vf_coef','ent_coef','lr','max_grad_norm'):
            value=getattr(self,name)
            if not math.isfinite(value) or value<0:raise ValueError(f'{name} must be finite and nonnegative')
        if not 0<self.gamma<=1 or not 0<=self.gae_lambda<=1:raise ValueError('Invalid gamma or GAE lambda')
        if self.lr<=0 or self.max_grad_norm<=0:raise ValueError('Learning rate and gradient norm must be positive')
        if not math.isfinite(self.policy_temperature) or self.policy_temperature<=0:
            raise ValueError('policy_temperature must be finite and strictly positive')
        if self.lr_horizon_timesteps is not None and (not isinstance(self.lr_horizon_timesteps,int) or self.lr_horizon_timesteps<=0):
            raise ValueError('lr_horizon_timesteps must be a positive integer')
        if not isinstance(self.map_view_size,int) or isinstance(self.map_view_size,bool) or self.map_view_size<9 or self.map_view_size%2!=1:
            raise ValueError('map_view_size must be an odd integer >= 9')
        for name in ('ctec_coef','ctec_lr','ctec_gamma','ctec_fade_start','ctec_clip'):
            if not math.isfinite(getattr(self,name)) or getattr(self,name)<0:raise ValueError('Invalid '+name)
        if not 0<self.ctec_gamma<1 or not 0<=self.ctec_fade_start<1:raise ValueError('Invalid C-TeC horizon')
        if self.ctec_lr<=0 or self.ctec_clip<=0:raise ValueError('C-TeC learning rate/clip must be positive')
        for name in ('ctec_width','ctec_dim','ctec_batch','ctec_updates','ctec_ramp_steps'):
            if type(getattr(self,name)) is not int or getattr(self,name)<1:raise ValueError('Invalid '+name)
        if type(self.ctec_warmup_steps) is not int or self.ctec_warmup_steps<0:raise ValueError('Invalid C-TeC warmup')
        if not 0<=self.recovery_probability<=1 or bool(self.recovery_bank)!=(self.recovery_probability>0):
            raise ValueError('Configure recovery_bank together with a probability in (0,1]')
        if self.recovery_bank and not Path(self.recovery_bank).is_file():
            raise ValueError('Recovery bank missing')
        if self.reset_seed_bank:
            data=json.loads(Path(self.reset_seed_bank).read_text())
            seeds=data['world_seeds'] if isinstance(data,dict) else data
            if not seeds or not isinstance(seeds,list) or any(not isinstance(s,int) or isinstance(s,bool) or not 0<=s<2**32 for s in seeds) or len(set(seeds))!=len(seeds):
                raise ValueError('reset_seed_bank must contain unique uint32 world seeds')


@dataclass(frozen=True)
class Budget:
    max_hours:float=9.
    evaluation_reserve_minutes:float=15.
    max_steps:int=10000000000

    def validate(self,batch):
        if not math.isfinite(self.max_hours) or self.max_hours<=0:raise ValueError('max_hours must be positive')
        if not math.isfinite(self.evaluation_reserve_minutes) or not 0<=self.evaluation_reserve_minutes<self.max_hours*60:
            raise ValueError('Evaluation reserve must be shorter than the total budget')
        if not isinstance(self.max_steps,int) or self.max_steps<batch:raise ValueError('max_steps must contain at least one PPO update')


@dataclass(frozen=True)
class Evaluation:
    common_checkpoints:dict=field(default_factory=lambda:{'100m':100000000,'1b':1000000000})
    common_episodes:int=1024
    final_episodes:int=256
    batch:int=32
    seed:int=18050
    temperatures:tuple=(1.,)
    world_seeds_file:str|None=None
    repeats:int=1
    defer_until_training_complete:bool=False

    def validate(self,max_steps):
        if not isinstance(self.defer_until_training_complete,bool):raise ValueError('defer_until_training_complete must be boolean')
        if not isinstance(self.batch,int) or self.batch<=0:raise ValueError('Evaluation batch must be positive')
        for count in (self.common_episodes,self.final_episodes):
            if not isinstance(count,int) or count<1 or count%self.batch:raise ValueError('Evaluation episode counts must be positive multiples of batch')
        if not isinstance(self.seed,int) or not 0<=self.seed<2**32:raise ValueError('Invalid evaluation seed')
        if not self.temperatures or len(set(self.temperatures))!=len(self.temperatures) or any(not math.isfinite(t) or t<0 for t in self.temperatures):
            raise ValueError('Evaluation temperatures must be unique finite nonnegative values')
        if not isinstance(self.repeats,int) or self.repeats<1:raise ValueError('Evaluation repeats must be positive')
        if self.repeats!=1 and not self.world_seeds_file:raise ValueError('Repeated evaluation requires a world seed file')
        if self.world_seeds_file:
            data=json.loads(Path(self.world_seeds_file).read_text())
            seeds=data['world_seeds'] if isinstance(data,dict) else data
            if not isinstance(seeds,list) or not seeds or any(not isinstance(s,int) or isinstance(s,bool) or not 0<=s<2**32 for s in seeds) or len(set(seeds))!=len(seeds):
                raise ValueError('Evaluation world seeds must be unique uint32 integers')
            if len(seeds)*self.repeats!=self.final_episodes:
                raise ValueError('final_episodes must match evaluation worlds times repeats')
        for label,step in self.common_checkpoints.items():
            if not re.fullmatch('[a-z0-9_]+',label) or label=='wallclock':raise ValueError('Invalid/reserved evaluation label')
            if not isinstance(step,int) or not 0<step<=max_steps:raise ValueError('Common evaluation checkpoint exceeds max_steps')


@dataclass(frozen=True)
class Architecture:
    name:str
    train_gpu:int
    evaluation_gpu:int
    train_gpus:tuple=()

    def __post_init__(self):
        object.__setattr__(self,'train_gpus',tuple(self.train_gpus))

    @property
    def training_devices(self):
        return self.train_gpus or (self.train_gpu,)


@dataclass(frozen=True)
class Experiment:
    name:str
    seed:int=50
    resume_from:str|None=None
    resume_after:str|None=None
    resume_reset_optimizer:bool=False
    checkpoint_milestones:tuple=LOG_MILESTONES
    architectures:tuple=field(default_factory=lambda:tuple(Architecture(name,i,i+4) for i,name in enumerate(DEFAULT_ARCHITECTURES)))
    rewards:Rewards=field(default_factory=Rewards)
    ppo:PPO=field(default_factory=PPO)
    budget:Budget=field(default_factory=Budget)
    evaluation:Evaluation=field(default_factory=Evaluation)
    wandb_project:str='Craftax_Baselines'
    wandb_entity:str='synaderi-uc-davis'

    @classmethod
    def load(cls,path):
        data=json.loads(Path(path).read_text())
        for key,kind in [('rewards',Rewards),('ppo',PPO),('budget',Budget),('evaluation',Evaluation)]:
            if key in data:data[key]=construct(kind,data[key])
        if 'architectures' in data:
            jobs=[]
            for item in data['architectures']:
                item=dict(item)
                if 'train_gpu' not in item and item.get('train_gpus'):
                    item['train_gpu']=item['train_gpus'][0]
                jobs.append(construct(Architecture,item))
            data['architectures']=tuple(jobs)
        if 'checkpoint_milestones' in data:data['checkpoint_milestones']=tuple(data['checkpoint_milestones'])
        result=construct(cls,data);result.validate();return result

    @property
    def batch_steps(self):return self.ppo.num_envs*self.ppo.num_steps

    def validate(self):
        if not re.fullmatch('[a-zA-Z0-9][a-zA-Z0-9_-]{0,95}',self.name):raise ValueError('Use a simple study name without slashes or spaces')
        if self.rewards.map_coverage_bonus is not None and any(j.name != 'map_mlp' for j in self.architectures):
            raise ValueError('Map coverage reward requires the feedforward map_mlp trainer')
        if not isinstance(self.seed,int) or not 0<=self.seed<2**32:raise ValueError('Invalid training seed')
        if not self.architectures or len({x.name for x in self.architectures})!=len(self.architectures):raise ValueError('Architectures must be unique and nonempty')
        devices=[]
        for job in self.architectures:
            if job.name not in ARCHITECTURES:raise ValueError(f'Unknown architecture {job.name}')
            count=len(job.training_devices)
            if len(set(job.training_devices))!=count:raise ValueError('Training GPU IDs must be unique')
            if count not in (1,2,4):raise ValueError('Data-parallel training supports 1, 2, or 4 GPUs')
            if job.train_gpu!=job.training_devices[0]:raise ValueError('train_gpu must match the first train_gpus entry')
            if count>1:
                if job.name not in ('gru','spatial_gru','spatial_map_gru','map_mlp','map_mlp_gru','neural_map_gru'):
                    raise ValueError('Data-parallel training supports recurrent GRU and feedforward map_mlp models')
                if self.ppo.num_envs%count:
                    raise ValueError('Global environments must divide evenly across training GPUs')
                if job.name=='map_mlp' and self.batch_steps%(count*self.ppo.num_minibatches):
                    raise ValueError('Each global feedforward minibatch must divide evenly across training GPUs')
                if job.name!='map_mlp' and self.ppo.num_envs%(count*self.ppo.num_minibatches):
                    raise ValueError('Each global recurrent minibatch must divide evenly across training GPUs')
            for gpu in (*job.training_devices,job.evaluation_gpu):
                if not isinstance(gpu,int) or isinstance(gpu,bool) or gpu<0:raise ValueError('GPU IDs must be nonnegative integers')
            if job.evaluation_gpu in job.training_devices and not self.evaluation.defer_until_training_complete:
                raise ValueError('Sharing a training/evaluation GPU requires deferred evaluation')
            devices.extend(dict.fromkeys((*job.training_devices,job.evaluation_gpu)))
        if len(set(devices))!=len(devices):raise ValueError('Assign distinct GPUs to training and evaluation jobs')
        self.rewards.validate();self.ppo.validate();self.budget.validate(self.batch_steps)
        old_crop=self.ppo.map_warmstart_crop_from
        if old_crop is not None and not (
            type(old_crop) is int and 9<=old_crop<self.ppo.map_view_size and old_crop%2==1
            and self.resume_from and self.resume_reset_optimizer and not self.ppo.map_global_readout
            and all(job.name=='map_mlp' for job in self.architectures)):
            raise ValueError('Crop widening needs a smaller odd source crop and a reset-Adam plain map_mlp resume')
        if self.ppo.reset_gru_every_step and any(j.name not in ('gru','spatial_gru','spatial_map_gru','map_mlp_gru','neural_map_gru') for j in self.architectures):
            raise ValueError('reset_gru_every_step requires a GRU model')
        if self.ppo.gru_concat_observation and any(j.name not in ('gru','spatial_gru','spatial_map_gru','map_mlp_gru','neural_map_gru') for j in self.architectures):
            raise ValueError('gru_concat_observation requires a GRU model')
        if any((j.name=='neural_map_gru') != (self.ppo.neural_map_channels>0) for j in self.architectures):raise ValueError('neural_map_gru requires positive neural_map_channels, exclusive to this architecture')
        if self.ppo.neural_map_reset_every_step and not self.ppo.neural_map_channels:raise ValueError('Neural map ablation requires a neural map')
        if self.ppo.ctec_coef and any(j.name!='map_mlp_gru' for j in self.architectures):
            raise ValueError('C-TeC currently supports map_mlp_gru')
        if self.ppo.map_survival_features!='none' and any(j.name not in ('map_mlp_gru','neural_map_gru') for j in self.architectures):
            raise ValueError('Survival map features require map_mlp_gru')
        if self.ppo.map_global_readout and any(j.name!='map_mlp' for j in self.architectures):
            raise ValueError('Global map readout requires the feedforward map_mlp model')
        if self.ppo.map_v1_attention_adapter and not (self.ppo.map_v1_imitation_init or self.resume_from or self.resume_after):
            raise ValueError('Frozen V1 requires trained weights for its decision layers')
        if self.ppo.map_v1_imitation_init and (not self.ppo.map_v1_attention_adapter or self.resume_from or self.resume_after
                or not (Path(self.ppo.map_v1_imitation_init)/'_METADATA').is_file()):
            raise ValueError('Select a complete V1 imitation checkpoint for a fresh adapter PPO run')
        if self.ppo.world_view and (self.ppo.world_view not in ('terrain','entities','local') or not self.ppo.motion_inputs):
            raise ValueError('World view requires a local motion GRU and a supported field of view')
        if self.ppo.local_policy_init and (not self.ppo.world_view or self.resume_from or self.resume_after or not Path(self.ppo.local_policy_init).is_dir()):
            raise ValueError('Local-policy transfer needs a source checkpoint and a fresh world-view study')
        if self.ppo.motion_inputs and any(j.name!='gru' for j in self.architectures):
            raise ValueError('motion_inputs currently requires the flat symbolic GRU model')
        if self.ppo.sru_gru and not (self.ppo.motion_inputs and self.ppo.gru_concat_observation
                and (self.resume_from or self.resume_after or self.ppo.sru_residual_actor_critic) and not self.ppo.world_view
                and not self.ppo.reset_gru_every_step):
            raise ValueError('SRU continuation requires the regular motion-input GRU-concat policy')
        if self.resume_after:
            if self.resume_from or self.resume_reset_optimizer or len(self.architectures)!=1:
                raise ValueError('Queued resume requires one job, no explicit checkpoint, and no optimizer restart')
            if not (Path(self.resume_after)/'progress.json').exists():
                raise ValueError('resume_after must name a source architecture directory containing progress.json')
        if self.ppo.recovery_bank and any(j.name!='map_mlp' for j in self.architectures):
            raise ValueError('Recovery starts require the map-MLP state format')
        if self.ppo.reset_seed_bank and any(j.name not in ('mlp','spatial','map_mlp') for j in self.architectures):
            raise ValueError('Fixed-world training currently supports feedforward MLP/CNN')
        if self.resume_reset_optimizer and (not self.resume_from or any(j.name not in ('mlp','spatial','map_mlp') for j in self.architectures)):
            raise ValueError('Optimizer restart requires a feedforward resume checkpoint')
        if self.ppo.policy_temperature!=1. and any(j.name not in ('mlp','spatial','map_mlp') for j in self.architectures):
            raise ValueError('Training temperature currently supports feedforward models')
        if any(not isinstance(step,int) or step<1 for step in self.checkpoint_milestones):raise ValueError('Checkpoint milestones must be positive integers')
        if self.ppo.lr_horizon_timesteps is not None and self.ppo.lr_horizon_timesteps<self.budget.max_steps:
            raise ValueError('Learning-rate horizon must cover max_steps')
        if self.resume_from:
            checkpoint=Path(self.resume_from)
            if not checkpoint.name.isdigit():raise ValueError('resume_from must name an explicit numeric checkpoint directory')
            start=int(checkpoint.name)
            if start%self.batch_steps:raise ValueError('Resume checkpoint must align with the configured rollout batch')
            if self.budget.max_steps-start<self.batch_steps:raise ValueError('Resume target must include at least one new PPO update')
        actual_max=self.budget.max_steps//self.batch_steps*self.batch_steps
        self.evaluation.validate(actual_max)

    def to_dict(self):return asdict(self)
    def fingerprint(self):
        data=self.to_dict()
        if self.ppo.recovery_bank:data['recovery_bank_sha256']=hashlib.sha256(Path(self.ppo.recovery_bank).read_bytes()).hexdigest()
        if self.ppo.reset_seed_bank:data['reset_seed_bank_sha256']=hashlib.sha256(Path(self.ppo.reset_seed_bank).read_bytes()).hexdigest()
        if self.evaluation.world_seeds_file:data['evaluation_world_seeds_sha256']=hashlib.sha256(Path(self.evaluation.world_seeds_file).read_bytes()).hexdigest()
        return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()

    def metadata(self,job,root,python):
        if self.resume_after:raise ValueError('Resolve resume_after with --queue before preparing training')
        p=self.ppo;batch=self.batch_steps
        requested=sorted({x for x in self.checkpoint_milestones if x<self.budget.max_steps}|set(self.evaluation.common_checkpoints.values()))
        recurrent=job.name in ('gru','spatial_gru','spatial_map_gru','map_mlp_gru','neural_map_gru')
        encoder=('map_mlp' if job.name in ('map_mlp','map_mlp_gru','neural_map_gru') else 'spatial_map' if job.name=='spatial_map_gru' else 'spatial' if job.name.startswith('spatial') else 'mlp')
        command=[python,'-u',str(Path(root)/('ppo_rnn_og.py' if recurrent else 'ppo.py')),
            '--env_name','Craftax-Classic-Symbolic-v1','--total_timesteps',str(self.budget.max_steps),'--seed',str(self.seed)]
        for key,value in asdict(p).items():
            if key in ('map_global_readout','map_full_terrain','map_single_readout','map_board_only','map_board_attention','map_v1_attention_adapter','map_internal_crop','map_patch_mlp','map_spatial_regions','map_fine_shortcut'):
                if value:command.append('--'+key)
                continue
            if key.startswith('recovery_') and not p.recovery_bank:continue
            if key.startswith('neural_map_') and not p.neural_map_channels:continue
            if key=='neural_map_reset_every_step':
                command.append('--neural_map_reset_every_step' if value else '--no-neural_map_reset_every_step');continue
            if key=='map_survival_features' and value=='none':continue
            if key.startswith('ctec_') and not p.ctec_coef:continue
            if key in ('sru_gru','sru_failure_replay','sru_residual_actor_critic'):
                if value:command.append('--'+key)
                continue
            if key.startswith('sru_') and not p.sru_gru:continue
            if key in ('anneal_lr','optimistic_resets','reset_gru_every_step','gru_concat_observation','motion_inputs') or value is None:continue
            if key=='map_view_size' and encoder not in ('spatial_map','map_mlp'):continue
            if key=='map_embedding_size' and encoder!='map_mlp':continue
            if key=='policy_temperature' and recurrent:continue
            flag='optimistic_reset_ratio' if key=='optimistic_reset_ratio' else key
            command.extend(['--'+flag,str(value)])
        command.extend(['--anneal_lr' if p.anneal_lr else '--no-anneal_lr',
            '--use_optimistic_resets' if p.optimistic_resets else '--no-use_optimistic_resets',
            '--num_repeats','1','--symbolic_encoder',encoder])
        if recurrent:command.append('--reset_gru_every_step' if p.reset_gru_every_step else '--no-reset_gru_every_step')
        if recurrent:command.append('--gru_concat_observation' if p.gru_concat_observation else '--no-gru_concat_observation')
        if recurrent:command.append('--motion_inputs' if p.motion_inputs else '--no-motion_inputs')
        for key,value in asdict(self.rewards).items():
            if value is not None:command.extend(['--'+key,str(value)])
        command.extend(['--save_policy','--save_policy_milestones',','.join(map(str,requested)),
            '--use_wandb','--wandb_project',self.wandb_project,'--wandb_entity',self.wandb_entity,
            '--run_name',f'{self.name}-{job.name}-seed{self.seed}'])
        if self.resume_from:command.extend(['--resume_from',str(Path(self.resume_from).resolve())])
        if len(job.training_devices)>1:
            command.extend(['--data_parallel_devices',str(len(job.training_devices))])
        if self.resume_reset_optimizer:command.append('--resume_reset_optimizer')
        start=int(Path(self.resume_from).name) if self.resume_from else 0
        return dict(name=self.name,architecture=job.name,seed=self.seed,fresh_training=not bool(self.resume_from or p.local_policy_init),
            resume_from=str(Path(self.resume_from).resolve()) if self.resume_from else None,
            resume_reset_optimizer=self.resume_reset_optimizer,start_timesteps=start,
            learning_rate_horizon_steps=self.ppo.lr_horizon_timesteps or (self.budget.max_steps-start if self.resume_reset_optimizer else self.budget.max_steps),
            reset_seed_bank_sha256=hashlib.sha256(Path(self.ppo.reset_seed_bank).read_bytes()).hexdigest() if self.ppo.reset_seed_bank else None,
            train_gpu=job.train_gpu,train_gpus=list(job.training_devices),
            data_parallel_devices=len(job.training_devices),
            local_num_envs=p.num_envs//len(job.training_devices),
            evaluation_gpu=job.evaluation_gpu,command=command,
            requested_training_steps=self.budget.max_steps,actual_training_steps=None,
            environment='Craftax-Classic-Symbolic-v1',max_episode_steps=10000,
            symbolic_encoder=encoder,explicit_odometry=p.motion_inputs or encoder in ('spatial_map','map_mlp'),
            map_memory=('Full simulator terrain plus observed terrain memory' if p.map_full_terrain else
                'fixed observed board with trainable attention and frozen V1 decision layers' if p.map_v1_attention_adapter else
                'fixed observed world board with player-conditioned attention' if p.map_board_attention else
                'fixed observed world board with dated terrain, visible entities, and player attributes' if p.map_board_only else
                'observed world with relative coordinates, 8x8x32 regions, and full live observation' if p.map_spatial_regions else
                'single observed world map and live view, neighborhood read inside the policy' if p.map_internal_crop else
                'single observed world map and live view' if p.map_single_readout else
                'observed terrain only; episode-persistent backing map; '
                + ('agent-centered and global readouts' if p.map_global_readout else 'agent-centered readout'))
                if encoder in ('spatial_map','map_mlp') else None,
            recurrent=recurrent,additional_death_cost=-self.rewards.death_penalty,map_coverage_bonus=self.rewards.map_coverage_bonus,
            additional_lava_cost=-self.rewards.lava_penalty,first_diamond_total_reward=self.rewards.diamond_reward,
            first_eat_plant_total_reward=self.rewards.plant_reward,all22_terminal_bonus=self.rewards.all22_terminal_bonus,**asdict(p),
            requested_checkpoints=requested,
            evaluation=dict(common_checkpoints=[(label,(step+batch-1)//batch*batch) for label,step in self.evaluation.common_checkpoints.items()],
                episodes_per_common_checkpoint=self.evaluation.common_episodes,final_episodes=self.evaluation.final_episodes,
                batch=self.evaluation.batch,seed=self.evaluation.seed,temperatures=self.evaluation.temperatures,
                world_seeds_file=self.evaluation.world_seeds_file,repeats=self.evaluation.repeats,
                defer_until_training_complete=self.evaluation.defer_until_training_complete))
