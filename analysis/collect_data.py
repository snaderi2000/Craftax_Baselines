import argparse
import os
import yaml
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import flashbax as fbx
from ppo_rnn import ActorCriticRNN, ScannedRNN
from craftax.craftax_env import make_craftax_env_from_name

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_path", type=str, required=True, help="Path to wandb run folder")
    parser.add_argument("--timestep", type=int, default=1000000)
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=600)
    args = parser.parse_args()

    # 1. Load Config & Env
    config_path = os.path.join(args.run_path, "files/config.yaml")
    with open(config_path) as f:
        raw_config = yaml.load(f, Loader=yaml.Loader)
        cfg = {k: v["value"] if isinstance(v, dict) and "value" in v else v for k, v in raw_config.items()}

    env = make_craftax_env_from_name(cfg.get("ENV_NAME", "Craftax-Classic-Pixels-v1"), True)
    env_params = env.default_params

    # 2. Restore Model
    action_dim = env.action_space(env_params).n
    network = ActorCriticRNN(action_dim=action_dim, config=cfg)
    ckpt_path = os.path.abspath(os.path.join(args.run_path, "files/policies"))
    mngr = ocp.CheckpointManager(ckpt_path, ocp.PyTreeCheckpointer())
    trained_params = mngr.restore(args.timestep)['params']

    # 3. Define the Vmapped Scan Step
    def policy_step(carry, _):
        hstate, env_state, last_obs, rng, done_mask = carry
        
        # Inference (Actor + Critic)
        obs_in = last_obs[:, None, :] 
        done_in = done_mask[:, None, None]
        new_hstate, pi, value = network.apply(trained_params, hstate, (obs_in, done_in))
        new_hstate = new_hstate.squeeze(0)
        
        # Action Sampling
        rng, action_rng = jax.random.split(rng)
        action_rngs = jax.random.split(action_rng, args.num_episodes)
        action = jax.vmap(lambda p, r: p.sample(seed=r))(pi, action_rngs).squeeze()

        # Env Step
        rng, step_rng = jax.random.split(rng)
        step_rngs = jax.random.split(step_rng, args.num_episodes)
        obs, next_env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
            step_rngs, env_state, action, env_params
        )

        transition = {
            "obs": last_obs,
            "action": action,
            "reward": reward,
            "value": value.squeeze(),
            "next_obs": obs,
            "done": done
        }
        
        return (new_hstate, next_env_state, obs, rng, done), transition

    # 4. Run Collection
    rng = jax.random.PRNGKey(42)
    rng, reset_rng = jax.random.split(rng)
    reset_rngs = jax.random.split(reset_rng, args.num_episodes)
    obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rngs, env_params)
    init_hstate = ScannedRNN.initialize_carry(args.num_episodes, 256)

    if init_hstate.ndim == 2:
        init_hstate = init_hstate[None, :]

    init_hstate = init_hstate.squeeze(0) if init_hstate.ndim == 3 else init_hstate
    
    init_carry = (init_hstate, env_state, obs, rng, jnp.zeros(args.num_episodes, dtype=bool))
    
    print(f"Collecting {args.num_episodes} episodes...")
    _, trajectory = jax.lax.scan(policy_step, init_carry, None, length=args.max_steps)

    # 5. Save to Flashbax-compatible format
    # Reshape from (steps, batch, ...) to (total_transitions, ...)
    flat_traj = jax.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), trajectory)
    
    output_path = f"craftax_data_t{args.timestep}.npz"
    jnp.savez(output_path, **flat_traj)
    print(f"Saved {args.num_episodes * args.max_steps} transitions to {output_path}")

if __name__ == "__main__":
    main()