import d3rlpy
import numpy as np
import os

# ===============================================================
# PART 1: SETUP - Define the file name to load
# ===============================================================
file_name = "latest_2048_episodes.h5"

if not os.path.exists(file_name):
    print(f"❌ Error: Data file not found!")
    print(f"Please make sure '{file_name}' is in the correct directory.")
else:
    print(f"--- Found data file: '{file_name}' ---")

    # ===============================================================
    # PART 2: DEFINITION - Your CustomEpisode Class
    # ===============================================================
    print("--- Part 2: Defining the CustomEpisode class ---")

    class CustomEpisode(d3rlpy.dataset.Episode):
        def __init__(self, observations, actions, rewards, terminals, progress):
            is_episode_terminated = bool(terminals[-1])
            
            # The parent constructor now correctly sets the .terminated attribute
            super().__init__(
                observations=observations,
                actions=actions,
                rewards=rewards,
                terminated=is_episode_terminated,
            )
            
            # ✅ FIX: The redundant assignment below has been removed.
            # self.terminated = is_episode_terminated <--- REMOVED
            
            # Store our custom data arrays
            # We use object.__setattr__ because the instance is frozen
            object.__setattr__(self, 'terminals', np.asarray(terminals, dtype=np.float32))
            object.__setattr__(self, 'progress', np.asarray(progress, dtype=np.float32))

    print("✅ CustomEpisode class defined.\n")

    # ===============================================================
    # PART 3: TRANSFORMATION - Load and convert your episodes
    # ===============================================================
    print("--- Part 3: Loading your dataset and converting to CustomEpisode ---")

    with open(file_name, "rb") as f:
        buffer = d3rlpy.dataset.InfiniteBuffer()
        loaded_dataset = d3rlpy.dataset.ReplayBuffer.load(f, buffer=buffer)

    new_episodes_with_progress = []

    for episode in loaded_dataset.episodes:
        progress_data = np.linspace(
            0.0, 1.0, num=episode.size(), dtype=np.float32
        )
        terminals_data = np.zeros(episode.size(), dtype=np.float32)
        if episode.terminated:
            terminals_data[-1] = 1.0

        new_episode = CustomEpisode(
            observations=episode.observations,
            actions=episode.actions,
            rewards=episode.rewards,
            terminals=terminals_data,
            progress=progress_data,
        )
        new_episodes_with_progress.append(new_episode)

    print(f"✅ Successfully converted {len(new_episodes_with_progress)} episodes.\n")

    # ===============================================================
    # PART 4: VERIFICATION - Check if it worked on your data
    # ===============================================================
    print("--- Part 4: Verifying the first CustomEpisode from your data ---")

    if new_episodes_with_progress:
        first_custom_episode = new_episodes_with_progress[0]
        print(f"Object type is CustomEpisode: {isinstance(first_custom_episode, CustomEpisode)}")
        print(f"Episode length: {first_custom_episode.size()}")
        # .terminated is set by the parent class, so this will work
        print(f"Episode is terminated: {first_custom_episode.terminated}")
        print(f"Has '.progress' attribute: {hasattr(first_custom_episode, 'progress')}")
        print(f"First 5 progress values: {first_custom_episode.progress[:5]}")
    else:
        print("⚠️ No episodes were found or converted from the dataset.")
