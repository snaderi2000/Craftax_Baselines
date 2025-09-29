import jax
import numpy as np
from craftax.craftax_env import make_craftax_env_from_name
from d3rlpy import load_learnable


# from models.actor_critic import ActorCritic
# from orbax.checkpoint import PyTreeCheckpointer


# ===============================
# CONFIG
# ===============================
MODEL_PATH = "top_1000.d3"
ENV_NAME = "Craftax-Classic-Symbolic-v1"
EVAL_EPISODES = 100
SEED = 42  # reproducibility

# Correct achievement keys
ACHIEVEMENT_KEYS = [
    "Achievements/collect_coal",
    "Achievements/collect_diamond",
    "Achievements/collect_drink",
    "Achievements/collect_iron",
    "Achievements/collect_sapling",
    "Achievements/collect_stone",
    "Achievements/collect_wood",
    "Achievements/defeat_skeleton",
    "Achievements/defeat_zombie",
    "Achievements/eat_cow",
    "Achievements/eat_plant",
    "Achievements/make_iron_pickaxe",
    "Achievements/make_iron_sword",
    "Achievements/make_stone_pickaxe",
    "Achievements/make_stone_sword",
    "Achievements/make_wood_pickaxe",
    "Achievements/make_wood_sword",
    "Achievements/place_furnace",
    "Achievements/place_plant",
    "Achievements/place_stone",
    "Achievements/place_table",
    "Achievements/wake_up",
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

    # # 1. Initialize the network architecture to match the trained model.
    # #    This creates the "empty shell" of your model.
    # network = ActorCritic(action_dim=env.action_space(env_params).n, layer_width=512)


    # # 2. Create an Orbax checkpointer to handle the loading.
    # orbax_checkpointer = PyTreeCheckpointer()

    # # 3. Restore the entire saved training state from the directory.
    # restored_train_state = orbax_checkpointer.restore(MODEL_PATH)

    # # 4. Extract just the model weights ('params') for inference.
    # params = restored_train_state["params"]

    @jax.jit
    def predict_action(p, obs):
        # The network expects a batch dimension, so we add one with obs[None, ...]
        pi, _ = network.apply(p, obs[None, ...])
        
        # For evaluation, we deterministically select the best action (mode)
        action = pi.mode()
        
        # The output action also has a batch dimension, so we remove it
        return action.squeeze()


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
            obs_np = np.array(obs, dtype=np.float32).reshape(1, -1)  # shape (1, obs_dim)
            action = cql.predict(obs_np)[0]

            #action = predict_action(params, obs)

            # Step environment
            rng, rng_step = jax.random.split(rng)
            obs, state, reward, done, info = env.step(rng_step, state, action, env_params)

            total_reward += float(reward)
            steps += 1

            # Update achievements
            for key in ACHIEVEMENT_KEYS:
                if float(info[key]) > 0:
                    achievement_counts[key] += 1

           

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
        clean_name = key.replace("Achievements/", "")
        rate = 100.0 * achievement_counts[key] / EVAL_EPISODES
        print(f"{clean_name}: {rate:.1f}%")
    
        # Calculate Reward and Score
    # Extract the total counts for each achievement and convert them to percentage rates
    achievement_rates = np.array(list(achievement_counts.values())) / EVAL_EPISODES * 100
    
    # Calculate the Reward (Arithmetic Mean of percentages)
    average_reward = np.mean(achievement_rates)

    # Calculate the Score using the geometric mean formula
    score = np.exp(np.mean(np.log1p(achievement_rates))) - 1

    # Print the calculated metrics
    print("\n===== Calculated Metrics =====")
    print(f"Reward (Average Achievement): {average_reward:.2f}")
    print(f"Score: {score:.2f}")



if __name__ == "__main__":
    evaluate_model()
