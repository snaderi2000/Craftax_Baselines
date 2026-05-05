import time

import jax.numpy as jnp
import numpy as np
import wandb

batch_logs = {}
log_times = []


def create_log_dict(info, config):
    to_log = {
        "episode_return": info["returned_episode_returns"],
        "episode_length": info["returned_episode_lengths"],
    }
    for k, v in info.items():
        if k.startswith("cumulative_"):
            to_log[k] = v

    sum_achievements = 0
    achievement_rates = []

    for k, v in info.items():
        if "achievements" in k.lower() and not k.startswith("cumulative_"):
            to_log[k] = v
            achievement_rates.append(v) # collect for score
            sum_achievements += v / 100.0

    to_log["achievements"] = sum_achievements

    if len(achievement_rates) > 0:
        reward = sum_achievements / len(achievement_rates) * 100
        to_log["reward"] = reward

    if len(achievement_rates) > 0:
        rates = np.array(achievement_rates, dtype =np.float32)
        score = np.exp(np.mean(np.log1p(rates))) - 1
        to_log["score"] = score


    if config.get("TRAIN_ICM") or config.get("USE_RND"):
        to_log["intrinsic_reward"] = info["reward_i"]
        to_log["extrinsic_reward"] = info["reward_e"]

        if config.get("TRAIN_ICM"):
            to_log["icm_inverse_loss"] = info["icm_inverse_loss"]
            to_log["icm_forward_loss"] = info["icm_forward_loss"]
        elif config.get("USE_RND"):
            to_log["rnd_loss"] = info["rnd_loss"]

    return to_log


def batch_log(update_step, log, config):
    update_step = int(update_step)
    if update_step not in batch_logs:
        batch_logs[update_step] = []

    batch_logs[update_step].append(log)

    if len(batch_logs[update_step]) == config["NUM_REPEATS"]:
        agg_logs = {}
        # unified logs: collect keys from the first record
        all_keys = list(batch_logs[update_step][0].keys())
        for key in all_keys:
            agg = []
            if key in ["goal_heatmap"]:
                agg = [batch_logs[update_step][0].get(key)]
            else:
                for i in range(config["NUM_REPEATS"]):
                    val = batch_logs[update_step][i].get(key)
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        agg.append(val)

            if len(agg) > 0:
                if key.startswith("cumulative_"):
                    agg_logs[key] = np.mean(agg)
                elif key in [
                    "episode_length",
                    "episode_return",
                    "score",
                    "reward",
                    "achievements",
                    "cumulative_score",
                    "cumulative_reward",
                    "cumulative_episodes",
                    "wm/loss_total",
                    "wm/loss_obs",
                    "wm/loss_rewards",
                    "wm/loss_ends",
                    "ppo/value_loss_real",
                    "ppo/value_loss_imag",
                ]:
                    agg_logs[key] = np.mean(agg)
                else:
                    agg_logs[key] = np.array(agg)

        log_times.append(time.time())

        if config["DEBUG"]:
            if len(log_times) == 1:
                print("Started logging")
            elif len(log_times) > 1:
                dt = log_times[-1] - log_times[-2]
                steps_between_updates = (
                    config["NUM_STEPS"] * config["NUM_ENVS"] * config["NUM_REPEATS"]
                )
                sps = steps_between_updates / dt
                agg_logs["sps"] = sps

        wandb.log(agg_logs)
        # free memory for this step
        del batch_logs[update_step]


# removed log_wm_loss; unified logging handles wm/loss_total
