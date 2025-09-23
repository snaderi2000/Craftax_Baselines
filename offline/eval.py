import jax
import numpy as np
from craftax.craftax_env import make_craftax_env_from_name
from d3rlpy import load_learnable

# -------------------
# Config
# -------------------
MODEL_PATH = "cql_craftax_final.d3"
N_EVAL_EPISODES = 20
MAX_STEPS = 5000

# Achievement names (your list, in order)
ACHIEVEMENT_NAMES = [
    "Collect Coal",
    "Collect Diamond",
    "Collect Drink",
    "Collect Iron",
    "Collect Sapling",
    "Collect Stone",
    "Collect Wood",
    "Defeat Skeleton",
    "Defeat Zombie",
    "Eat Cow",
    "Eat Plant",
    "Make Iron Pickaxe",
    "Make Iron Sword",
    "Make Stone Pickaxe",
    "Make Stone Sword",
    "Make Wood Pickaxe",
    "Make Wood Sword",
    "Place Furnace",
    "Place Plant",
    "Place Stone",
    "Place Table",
    "Wake Up",
]
N_ACH = len(ACHIEVEMENT_NAMES)

def get_ach_vector(info):
    """
    Return a 1D float array of length N_ACH for achievements this step.
    Tries common keys; pads/truncates to N_ACH if shape differs.
    """
    vec = None
    for k in ["achievements", "achievement", "achievement_counts"]:
        if k in info:
            vec = np.array(info[k], dtype=np.float32).ravel()
            break
    if vec is None:
        vec = np.zeros(N_ACH, dtype=np.float32)

    # pad or trim to match our list length
    if vec.size < N_ACH:
        vec = np.pad(vec, (0, N_ACH - vec.size))
    elif vec.size > N_ACH:
        vec = vec[:N_ACH]
    return vec

def main():
    # ---- Env ----
    print("Initializing Craftax-Classic-Symbolic-v1 ...")
    rng = jax.random.PRNGKey(0)
    env = make_craftax_env_from_name("Craftax-Classic-Symbolic-v1", True)
    env_params = env.default_params
    obs_shape = env.observation_space(env_params).shape
    action_n = env.action_space(env_params).n
    print(f"Obs shape: {obs_shape} | Action space size: {action_n}")

    # ---- Model ----
    print(f"Loading model: {MODEL_PATH}")
    agent = load_learnable(MODEL_PATH)
    print("Model loaded.\n")

    total_rewards = []
    # success counter: how many episodes each achievement was achieved at least once
    ach_episode_success = np.zeros(N_ACH, dtype=np.float32)

    for ep in range(N_EVAL_EPISODES):
        rng, r0 = jax.random.split(rng)
        obs, state = env.reset(r0, env_params)
        ep_reward = 0.0
        done = False

        # track whether each achievement was hit at least once in this episode
        ep_hit = np.zeros(N_ACH, dtype=np.float32)

        for t in range(MAX_STEPS):
            obs_np = np.asarray(obs, dtype=np.float32).reshape(1, -1)
            action = int(agent.predict(obs_np)[0])

            rng, r1 = jax.random.split(rng)
            obs, state, reward, done, info = env.step(r1, state, action, env_params)
            ep_reward += float(reward)

            ach_vec = get_ach_vector(info)
            ep_hit = np.maximum(ep_hit, (ach_vec > 0).astype(np.float32))

            if done:
                break

        total_rewards.append(ep_reward)
        ach_episode_success += ep_hit

        print(f"Episode {ep+1}/{N_EVAL_EPISODES}: reward={ep_reward:.2f}, steps={t+1}")

    # ---- Summary ----
    print("\n===== FINAL EVALUATION =====")
    print(f"Episodes: {N_EVAL_EPISODES}")
    print(f"Average Reward: {np.mean(total_rewards):.2f}")

    print("\nAchievement Success Rates:")
    for i, name in enumerate(ACHIEVEMENT_NAMES):
        rate = (ach_episode_success[i] / N_EVAL_EPISODES) * 100.0
        print(f"{name} {rate:.1f}%")

    mean_rate = np.mean(ach_episode_success / N_EVAL_EPISODES) * 100.0
    print(f"\nOverall Mean Success Rate: {mean_rate:.1f}%")

if __name__ == "__main__":
    main()
