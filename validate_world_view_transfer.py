"""Verify a local motion checkpoint on factual Classic states and its widened inputs."""
import argparse,json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from craftax.craftax_env import make_craftax_env_from_name
from ppo_rnn_og import ActorCriticRNN, initialize_recurrent_carry
from symbolic_world_view import LOCAL, augment, source_params, transplant
from symbolic_motion_inputs import with_motion_inputs


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-study',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    cfg=json.loads((args.source_study/'inference_config.json').read_text())
    checkpoint=json.loads((args.source_study/'evaluation_wallclock/summary.json').read_text())['checkpoint']
    source=source_params(checkpoint)
    env=with_motion_inputs(make_craftax_env_from_name(cfg['ENV_NAME'],False),cfg)
    key=jax.random.PRNGKey(34455)
    obs,state=env.reset(key,env.default_params)
    modes=('local','terrain','entities')
    sequence={m:[] for m in modes}
    for t in range(8):
        for mode in modes:
            x=augment(obs,state,mode)
            np.testing.assert_array_equal(x[:LOCAL],obs)
            board=np.asarray(x[LOCAL:-2],np.int32).reshape(64,64)
            raw=state.env_state
            if mode!='local':
                np.testing.assert_array_equal((board & 31)-1,raw.map)
            else:
                radius=np.indices((64,64))-np.asarray(raw.player_position)[:,None,None]
                expected=(abs(radius[0])<=3)&(abs(radius[1])<=4)
                np.testing.assert_array_equal(board>0,expected)
            if mode=='entities':
                for i,mobs in enumerate((raw.zombies,raw.cows,raw.skeletons,raw.arrows)):
                    expected=np.zeros((64,64),bool)
                    positions=np.asarray(mobs.position)[np.asarray(mobs.mask,bool)]
                    expected[positions[:,0],positions[:,1]]=True
                    np.testing.assert_array_equal(((board>>(5+i))&1)>0,expected)
            sequence[mode].append(x)
        obs,state,*_=env.step(jax.random.fold_in(key,t),state,1+t%4,env.default_params)
    inputs=jnp.stack(sequence['local'])[:,None]
    dones=jnp.zeros((8,1),bool).at[4,0].set(True)
    carry=initialize_recurrent_carry(1,cfg)
    base=ActorCriticRNN(17,cfg)
    h,pi,v=base.apply(source,carry,(inputs[...,:LOCAL],dones))
    result={}
    for mode in modes:
        net=ActorCriticRNN(17,dict(cfg,WORLD_VIEW=mode))
        x=jnp.stack(sequence[mode])[:,None]
        params=transplant(net.init(key,carry,(x[:1],dones[:1])),source)
        hh,p,vv=net.apply(params,carry,(x,dones))
        diffs=[float(jnp.max(jnp.abs(a-b))) for a,b in ((hh,h),(pi.logits,p.logits),(v,vv))]
        if max(diffs)>1e-5:raise ValueError(f'Transfer changed outputs: {diffs}')
        result[mode]=dict(hidden_max_error=diffs[0],logit_max_error=diffs[1],value_max_error=diffs[2])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(dict(source=checkpoint,steps=8,transfer=result),indent=2)+'\n')
    print(result)


if __name__=='__main__':main()
