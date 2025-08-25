# dataset_from_vault.py
import jax.numpy as jnp
import flashbax as fbx
from flashbax.vault import Vault

def make_replay_samplers(
    vault_uid: str,
    rel_dir: str = "/home/synaderi/Craftax_Baselines",
    T_wm: int = 20,
    batch_envs: int = 48,           # how many windows per batch
    train_pct=(0, 90),
    val_pct=(90, 100),
):
    # Open vault and read splits
    v = Vault("craftax_replay_buffer", rel_dir=rel_dir, vault_uid=vault_uid)
    D_train = v.read(percentiles=train_pct)
    D_val   = v.read(percentiles=val_pct)

    B, T_train = D_train.experience["actions"].shape
    T_val = D_val.experience["actions"].shape[1]

    # Build samplers (pure functions) over fixed arrays
    sampler_train = fbx.make_trajectory_buffer(
        max_length_time_axis=T_train,
        min_length_time_axis=T_wm,
        add_batch_size=B,
        sample_batch_size=batch_envs,
        sample_sequence_length=T_wm,
        period=1,
    )
    sampler_val = fbx.make_trajectory_buffer(
        max_length_time_axis=T_val,
        min_length_time_axis=T_wm,
        add_batch_size=B,
        sample_batch_size=batch_envs,
        sample_sequence_length=T_wm,
        period=1,
    )

    def _to_plain_dict(sample_dataclass):
        """Flashbax returns a dataclass with `.experience`; turn it into a plain dict."""
        exp = sample_dataclass.experience
        # exp is a pytree/dict-like; ensure we return a real dict with desired dtypes
        out = {
            "obs":     exp["obs"].astype(jnp.float32) / 255.0,  # (N, T_wm, 63,63,3)
            "actions": exp["actions"],                          # (N, T_wm)
            "rewards": exp["rewards"],                          # (N, T_wm)
            "dones":   exp["dones"],                            # (N, T_wm)
        }
        return out

    def sample_train(rng_key):
        sample = sampler_train.sample(D_train, rng_key=rng_key)
        return _to_plain_dict(sample)

    def sample_val(rng_key):
        sample = sampler_val.sample(D_val, rng_key=rng_key)
        return _to_plain_dict(sample)

    info = {"B": B, "T_train": T_train, "T_val": T_val}
    return sample_train, sample_val, info
