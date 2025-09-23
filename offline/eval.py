import jax
import numpy as np
from craftax.craftax_env import make_craftax_env_from_name
from d3rlpy import load_learnable

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "cql_craftax_final.d3"
ENV_NAME = "Craftax-Classic-Symbolic-v1"
EVAL_EPISODES = 20
SEED = 42  # reproducibility

# Correct achievement keys
ACHIEVEMENT_KEYS = [
    "collect_coal",
    "collect_diamond",
    "collect_drink",
    "collect_iron",
    "collect_sapling",
    "collect_stone",
    "collect_wood",
    "defeat_skeleton",
    "defeat_zombie",
    "eat_cow",
    "eat_plant",
    "make_iron_pickaxe",
    "make_iron_sword",
    "make_stone_pickaxe",
    "make_stone_sword",
    "make_wood_pickaxe",
    "make_wood_sword",
    "place_furnace",
    "place_plant",
    "place_stone",
    "place_table",
    "wake_up",
]

# ===============================
# EVALUATION
# ===============================
def evaluate_model():
    print(f"Initializing {ENV_NAME} ...")
    env = make_craftax_env_from_name(ENV_NAME, auto_reset=True)
    env_params = env.default_params

    rng = jax.random.PRNGKey(SEED)

    # Load trained model
    print(f"Loading model: {MODEL_PATH}")
    cql = load_learnable(MODEL_PATH)
    print("Model loaded.")

    # Track achievements
    achievement_counts = {key: 0 for key in ACHIEVEMENT_KEYS}

    episode_rewards = []

    for ep in range(EVAL_EPISODES):
        rng, rng_reset = jax.random.split(rng)
        obs, state = env.reset(rng_reset, env_params)

        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            # Convert obs to numpy for the model
            action = cql.predict([np.array(obs, dtype=np.float32)])[0]

            # Step environment
            rng, rng_step = jax.random.split(rng)
            obs, state, reward, done, info = env.step(rng_step, state, action, env_params)

            total_reward += float(reward)
            steps += 1

            # Update achievements
            if "achievements" in info:
                for key in ACHIEVEMENT_KEYS:
                    achievement_counts[key] += info["achievements"].get(key, 0)

        episode_rewards.append(total_reward)
        print(f"Episode {ep + 1}/{EVAL_EPISODES}: reward={total_reward:.2f}, steps={steps}")

    # ===============================
    # FINAL SUCCESS RATES
    # ===============================
    print("\n===== FINAL EVALUATION =====")
    print(f"Episodes: {EVAL_EPISODES}")
    print(f"Average Reward: {np.mean(episode_rewards):.2f}\n")

    print("Achievement Success Rates:")
    for key in ACHIEVEMENT_KEYS:
        rate = (achievement_counts[key] / EVAL_EPISODES) * 100
        print(f"{key}: {rate:.1f}%")

if __name__ == "__main__":
    evaluate_model()
