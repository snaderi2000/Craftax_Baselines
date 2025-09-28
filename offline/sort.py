import h5py
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# ---------- CONFIG ----------
INPUT_H5 = "1M_combined_dataset_compressed.h5"
OUTPUT_H5 = "sorted_dataset.h5"
BATCH_SIZE = 32768  # Increase if you have enough RAM (e.g., 16K or 32K)
# ----------------------------

def build_metadata(in_f):
    """
    Pass 1: Scan all shards and build metadata for sorting.
    Metadata contains (episode_id, step_in_episode, shard_id, row_id).
    """
    shard_names = list(in_f.keys())
    print(f"Found {len(shard_names)} shards")

    metadata_parts = []
    for shard_idx, shard_name in enumerate(tqdm(shard_names, desc="Scanning shards")):
        g = in_f[shard_name]
        ep_ids = g["episode_id"][:]            # shape (65536,)
        steps = g["step_in_episode"][:]        # shape (65536,)

        n = len(ep_ids)
        shard_indices = np.arange(n, dtype=np.int32)
        shard_column = np.full(n, shard_idx, dtype=np.int32)

        shard_meta = np.rec.fromarrays(
            [ep_ids, steps, shard_column, shard_indices],
            names=("ep", "step", "shard", "idx")
        )
        metadata_parts.append(shard_meta)

    metadata = np.concatenate(metadata_parts, axis=0)
    print(f"Total transitions: {len(metadata):,}")
    return metadata, shard_names


def create_output_file(out_f, template_group, total_transitions):
    """
    Create the structure of the sorted output HDF5 file.
    """
    for key, ds in template_group.items():
        if key == "info":
            continue  # skip singleton metadata
        shape = (total_transitions,) + ds.shape[1:]
        if key in {"obs", "next_obs"}:
            out_f.create_dataset(
                key,
                shape=shape,
                dtype=ds.dtype,
                compression="gzip",
                compression_opts=4
            )
        else:
            out_f.create_dataset(key, shape=shape, dtype=ds.dtype)


def copy_batch(batch, in_f, out_f, shard_names, batch_start):
    """
    Copy a batch of transitions efficiently by grouping reads by shard.
    """
    grouped = defaultdict(list)

    # Group by shard
    for out_pos, row in enumerate(batch):
        grouped[row["shard"]].append((row["idx"], out_pos))

    # Process shard-by-shard
    for shard_idx, idx_pairs in grouped.items():
        shard_name = shard_names[shard_idx]
        g = in_f[shard_name]

        # Sort indices inside this shard for contiguous read
        idx_pairs.sort(key=lambda x: x[0])
        read_indices = [p[0] for p in idx_pairs]
        write_positions = [p[1] for p in idx_pairs]

        write_positions = np.array(write_positions)

        # Copy each dataset
        for key in out_f.keys():
            data_block = g[key][read_indices]  # bulk read
            out_f[key][batch_start + write_positions] = data_block


def sort_combined_h5(input_file, output_file, batch_size=BATCH_SIZE):
    print(f"Opening input file: {input_file}")
    with h5py.File(input_file, "r") as in_f:
        # Step 1. Build metadata
        metadata, shard_names = build_metadata(in_f)

        # Step 2. Sort metadata
        print("Sorting metadata by episode_id, step_in_episode...")
        metadata.sort(order=["ep", "step"])
        total_transitions = len(metadata)
        print("Metadata sorted!")

        # Step 3. Create output file
        print(f"Creating output file: {output_file}")
        with h5py.File(output_file, "w") as out_f:
            create_output_file(out_f, in_f[shard_names[0]], total_transitions)

            # Step 4. Stream sorted data in batches
            print("Streaming data into sorted file...")
            for start in tqdm(range(0, total_transitions, batch_size), desc="Writing batches"):
                end = min(start + batch_size, total_transitions)
                batch = metadata[start:end]
                copy_batch(batch, in_f, out_f, shard_names, batch_start=start)

    print(f"\n✅ Done! Sorted dataset saved at: {output_file}")


if __name__ == "__main__":
    sort_combined_h5(INPUT_H5, OUTPUT_H5)

