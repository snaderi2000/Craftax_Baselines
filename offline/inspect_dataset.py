import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse

def inspect_dataset(file_path, plot=True):
    print("="*60)
    print(f" Inspecting dataset: {file_path}")
    print("="*60)

    with h5py.File(file_path, "r") as f:
        # -------------------------------
        # 1. Basic structure
        # -------------------------------
        print("\n[1] Dataset keys and shapes")
        keys = list(f.keys())
        print("Keys in file:", keys)
        n = len(f["done"])
        for k in keys:
            print(f"  {k}: shape={f[k].shape}, dtype={f[k].dtype}")
            assert f[k].shape[0] == n, f"Mismatch in {k} length!"
        print("All datasets aligned ✅")

        # -------------------------------
        # 2. Episode boundaries
        # -------------------------------
        done = np.array(f["done"])
        print("\n[2] Episode Boundaries")
        print(f"First done flag: {done[0]}, Last done flag: {done[-1]}")
        print("Total episodes:", done.sum())

        if not done[0] or not done[-1]:
            print("⚠️ Warning: Dataset does not start or end on a clean episode boundary!")

        # -------------------------------
        # 3. Episode length statistics
        # -------------------------------
        done_indices = np.where(done)[0]
        episode_lengths = np.diff(done_indices, prepend=-1)
        print("\n[3] Episode Length Stats")
        print(f"  Min length: {episode_lengths.min()}")
        print(f"  Max length: {episode_lengths.max()}")
        print(f"  Mean length: {episode_lengths.mean():.2f}")

        # -------------------------------
        # 4. Game phase verification
        # -------------------------------
        phases = np.array(f["game_phase"])
        print("\n[4] Game Phase Stats")
        unique_phases = set(phases)
        print("  Unique phase labels:", unique_phases)
        counts = {p: (phases == p).sum() for p in [0, 1, 2]}
        print("  Counts:", counts)

        if unique_phases - {0, 1, 2}:
            print("⚠️ Warning: Found unexpected game phase labels!")

        if plot:
            plt.figure()
            plt.hist(phases, bins=[-0.5, 0.5, 1.5, 2.5], rwidth=0.8)
            plt.xticks([0, 1, 2], ["Early", "Mid", "Late"])
            plt.title("Game Phase Distribution")
            plt.show()

        # -------------------------------
        # 5. Reward distribution
        # -------------------------------
        rewards = np.array(f["reward"])
        print("\n[5] Reward Stats")
        print(f"  Min: {rewards.min()}, Max: {rewards.max()}, Mean: {rewards.mean():.4f}")

        if plot:
            plt.figure()
            plt.hist(rewards, bins=100)
            plt.title("Reward Distribution")
            plt.show()

        # -------------------------------
        # 6. Verify obs and next_obs alignment
        # -------------------------------
        print("\n[6] Checking obs / next_obs alignment")
        obs_sample = f["obs"][:1000]
        next_obs_sample = f["next_obs"][:999]
        mismatch = np.sum(np.any(obs_sample[1:] != next_obs_sample, axis=1))
        print("  Mismatch count in first 1000 steps:", mismatch)
        if mismatch > 10:
            print("⚠️ Warning: Potential misalignment between obs and next_obs")

        # -------------------------------
        # 7. Check a single episode
        # -------------------------------
        ep_start = 0
        ep_end = done_indices[0]
        ep_phases = phases[ep_start:ep_end+1]
        ep_rewards = rewards[ep_start:ep_end+1]

        print("\n[7] First Episode Inspection")
        print("  Episode length:", ep_end - ep_start + 1)
        print("  Phase progression:", ep_phases[:10], "...", ep_phases[-10:])
        print("  Total episode reward:", ep_rewards.sum())

        # -------------------------------
        # 8. Final summary
        # -------------------------------
        summary = {
            "total_transitions": n,
            "total_episodes": done.sum(),
            "mean_episode_length": float(episode_lengths.mean()),
            "phase_distribution": counts
        }

        print("\n[8] Final Summary")
        for k, v in summary.items():
            print(f"  {k}: {v}")

    print("\nInspection complete ✅")

# -------------------------------------------------------
# CLI usage
# -------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a Craftax HDF5 dataset")
    parser.add_argument("--file", type=str, required=True, help="Path to .h5 file to inspect")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    args = parser.parse_args()

    inspect_dataset(args.file, plot=not args.no_plot)
