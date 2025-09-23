import wandb
from d3rlpy.dataset import ReplayBuffer, FIFOBuffer
from d3rlpy.algos import CQLConfig
from d3rlpy import load_learnable

# ---- W&B ----
wandb.init(project="Craftax-OfflineRL", name="cql_run_1")

# ---- Load Dataset ----
buffer = FIFOBuffer(limit=None)
with open("craftax_dataset_cleaned.h5", "rb") as f:
    replay_buffer = ReplayBuffer.load(f, buffer)

print(f"Dataset loaded: {len(replay_buffer.episodes)} episodes, {replay_buffer.transition_count} transitions")

# ---- Initialize CQL ----
cql = CQLConfig().create(device="cuda:0")

# ---- Train ----
cql.fit(
    replay_buffer,
    n_steps=500000,
    experiment_name="craftax_cql",
    with_timestamp=False,
    show_progress=True
)

# ---- Save Model ----
cql.save("cql_craftax_final.d3")
print("Training complete! Model saved to cql_craftax_final.d3")
