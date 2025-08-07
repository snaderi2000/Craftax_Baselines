#!/usr/bin/env python3
import argparse
import os
import yaml
import numpy as np
import jax
import jax.numpy as jnp
import optax
import imageio
from flax.training.train_state import TrainState
from orbax.checkpoint import PyTreeCheckpointer, CheckpointManager, CheckpointManagerOptions

# use your local wrappers import
from wrappers import AutoResetEnvWrapper
from models.actor_critic import ActorCritic, ActorCriticConv, ActorCriticConvRNN

def load_config(path):
    with open(path) as f:
        raw = yaml.load(f, Loader=yaml.Loader)
    cfg = {}
    for k,v in raw.items():
        if isinstance(v, dict) and "value" in v:
            cfg[k] = v["value"]
    return cfg

def make_env_and_model(config):
    name = config["ENV_NAME"]
    # pick env & net
    if name.endswith("Pixels-v1"):
        # pixel env
        if "Classic" in name:
            from craftax.craftax_classic.envs.craftax_pixels_env import CraftaxClassicPixelsEnv
            from craftax.craftax_classic.constants import Action
            env = CraftaxClassicPixelsEnv(CraftaxClassicPixelsEnv.default_static_params())
        else:
            from craftax.craftax.envs.craftax_pixels_env import CraftaxPixelsEnv
            from craftax.craftax.constants import Action
            env = CraftaxPixelsEnv(CraftaxPixelsEnv.default_static_params())
        network = ActorCriticConvRNN(
            action_dim = len(Action),
            head_width = config["LAYER_SIZE"],
            rnn_hidden = config["RNN_HIDDEN"],
            use_gru    = config["USE_GRU"],
            train      = False,     # <- IMPORTANT: eval mode
        )
    else:
        # symbolic env
        if "Classic" in name:
            from craftax.craftax_classic.envs.craftax_symbolic_env import CraftaxClassicSymbolicEnv
            from craftax.craftax_classic.constants import Action
            env = CraftaxClassicSymbolicEnv(CraftaxClassicSymbolicEnv.default_static_params())
        else:
            from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv
            from craftax.craftax.constants import Action
            env = CraftaxSymbolicEnv(CraftaxSymbolicEnv.default_static_params())
        network = ActorCritic(len(Action), config["LAYER_SIZE"])
    # wrap
    env = AutoResetEnvWrapper(env)
    return env, network

def restore_ckpt(ckpt_path, template_items):
    cp = PyTreeCheckpointer()

    # if they pointed at the “default” shard folder, load directly:
    if os.path.basename(ckpt_path) == "default":
        return cp.restore(ckpt_path, template_items)

    # else: they gave the `<step>` directory, so use a manager
    step = int(os.path.basename(ckpt_path))
    mgr = CheckpointManager(
        ckpt_path,
        cp,
        CheckpointManagerOptions(max_to_keep=1, create=False),
    )
    return mgr.restore(step, template_items)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path",    required=True,
                   help="…/files/policies/1000000/default")
    p.add_argument("--config_path",  required=True,
                   help="…/files/config.yaml")
    p.add_argument("--out_dir",      default="demos")
    p.add_argument("--num_episodes", type=int, default=10)
    p.add_argument("--fps",          type=int, default=20)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = load_config(args.config_path)
    cfg["NUM_ENVS"] = 1  # single
    env, net = make_env_and_model(cfg)
    env_params = env.default_params

    # dummy init to get shapes
    rng = jax.random.PRNGKey(0)
    rng, init_key = jax.random.split(rng)
    obs0, st0 = env.reset(init_key, env_params)

    # initialize empty RNN state if you have one:
    h = jnp.zeros((1, cfg["RNN_HIDDEN"]))

    # Get initial variables for model structure and shapes
    initial_model_variables = net.init(init_key, obs0, h)
    initial_params = initial_model_variables["params"]
    initial_batch_stats = initial_model_variables.get("batch_stats") # Use .get for safety

    # Construct a template for checkpoint restoration
    ckpt_template = {'params': initial_params}
    if initial_batch_stats is not None:
        ckpt_template['batch_stats'] = initial_batch_stats

    # build a flax TrainState just to hold params
    tx = optax.identity()
    state = TrainState.create(apply_fn=net.apply, params=initial_params, tx=tx)

    # restore checkpoint items
    restored_ckpt_items = restore_ckpt(args.ckpt_path, ckpt_template)

    # Update TrainState with restored parameters
    state = state.replace(params=restored_ckpt_items['params'])

    # Extract restored batch stats, will be None if not in checkpoint
    restored_batch_stats = restored_ckpt_items.get('batch_stats')

    # run episodes
    for epi in range(1, args.num_episodes+1):
        print(f"Episode {epi}/{args.num_episodes}")
        frames = []
        rng, key = jax.random.split(rng)
        obs, env_state = env.reset(key, env_params)
        done = False

        while not bool(done):
            # record raw pixels (assume obs is [H,W,C] uint8)
            frame = np.array(obs)
            frames.append(frame)

            # select action
            # Create variables dictionary for network.apply
            variables = {'params': state.params}
            if restored_batch_stats is not None:
                variables['batch_stats'] = restored_batch_stats

            obs_batch = jnp.expand_dims(obs, 0)   # (1, H, W, C)
            pi, value, h = net.apply(
                variables,
                obs_batch,
                h,
            )
            rng, key = jax.random.split(rng)
            a = np.int32(pi.sample(seed=key)[0])

            # step
            rng, key = jax.random.split(rng)
            obs, env_state, reward, done, info = env.step(key, env_state, a, env_params)

        # write mp4
        path = os.path.join(args.out_dir, f"episode_{epi:02d}.mp4")
        writer = imageio.get_writer(path, fps=args.fps)
        for im in frames:
            writer.append_data(im)
        writer.close()
        print(f" → wrote {path}")

if __name__=="__main__":
    main()
