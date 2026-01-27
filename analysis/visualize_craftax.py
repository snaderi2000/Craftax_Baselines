import argparse
import os
import yaml
import jax
import jax.numpy as jnp
import numpy as np
import imageio
import orbax.checkpoint as ocp

from ppo_rnn import ActorCriticRNN, ScannedRNN
from craftax.craftax_env import make_craftax_env_from_name

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_path", type=str, required=True)
    parser.add_argument("--timestep", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=42, help="Seed for env and action sampling")
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--upscale", type=int, default=4)
    args = parser.parse_args()

    # 1. Load Config
    config_path = os.path.join(args.run_path, "files/config.yaml")
    with open(config_path) as f:
        raw_config = yaml.load(f, Loader=yaml.Loader)
        cfg = {k: v["value"] if isinstance(v, dict) and "value" in v else v for k, v in raw_config.items()}

    # 2. Setup Environment
    env_name = cfg.get("ENV_NAME", "Craftax-Classic-Pixels-v1")
    env = make_craftax_env_from_name(env_name, True)
    env_params = env.default_params

    # 3. Initialize Model
    action_dim = env.action_space(env_params).n
    network = ActorCriticRNN(action_dim=action_dim, config=cfg)
    
    # 4. Restore Checkpoint
    ckpt_path = os.path.abspath(os.path.join(args.run_path, "files/policies"))
    mngr = ocp.CheckpointManager(ckpt_path, ocp.PyTreeCheckpointer())
    raw_data = mngr.restore(args.timestep)
    trained_params = raw_data['params']

    # 5. Rollout with Seed
    frames = []
    
    # Create the master RNG key from the provided seed
    rng = jax.random.PRNGKey(args.seed)
    rng, env_rng = jax.random.split(rng)
    
    # Reset env with the seed-derived key
    obs, env_state = env.reset(env_rng, env_params)
    hstate = ScannedRNN.initialize_carry(1, 256)
    done = False

    print(f"Starting rollout (Seed: {args.seed}) for {args.max_steps} steps...")
    
    for i in range(args.max_steps):
        # Frame Processing
        frame = np.array(obs)
        frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
        if args.upscale > 1:
            frame = frame.repeat(args.upscale, axis=0).repeat(args.upscale, axis=1)
        frames.append(frame)

        # Inference
        obs_in = obs[None, None, :]
        done_in = jnp.array([[done]])
        hstate, pi, value = network.apply(trained_params, hstate, (obs_in, done_in))
        
        # Step with Seeded Action
        rng, step_rng = jax.random.split(rng)
        action = pi.sample(seed=step_rng).squeeze()
        
        obs, env_state, reward, step_done, info = env.step(step_rng, env_state, action, env_params)
        
        done = bool(step_done)
        if done:
            print(f"Episode finished at step {i}")
            break

    # 6. Save
    output_name = f"vis_{env_name}_s{args.seed}_t{args.timestep}.mp4"
    imageio.mimsave(output_name, frames, fps=15)
    print(f"Success! Saved to {output_name}")

if __name__ == "__main__":
    main()