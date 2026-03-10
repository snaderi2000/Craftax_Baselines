import numpy as np
import jax
import jax.numpy as jnp

from token_wm.twm.world_model import WorldModel
from token_wm.twm.transformer import TransformerConfig


def _build_model(obs_loss_mode: str, ar_aux_weight: float = 0.25):
    cfg = TransformerConfig(
        tokens_per_block=3,   # K=2 obs tokens + 1 action token
        max_blocks=8,
        attention="block_causal",
        num_layers=1,
        num_heads=1,
        embed_dim=8,
        embed_pdrop=0.0,
        resid_pdrop=0.0,
        attn_pdrop=0.0,
    )
    model = WorldModel(
        obs_vocab_size=7,
        act_vocab_size=5,
        config=cfg,
        reward_num_classes=2,
        obs_loss_mode=obs_loss_mode,
        ar_aux_weight=ar_aux_weight,
    )
    dummy_tokens = jnp.zeros((1, cfg.tokens_per_block * 2), dtype=jnp.int32)
    params = model.init(jax.random.PRNGKey(0), dummy_tokens)
    params = jax.tree.map(jnp.zeros_like, params)
    return model, params


def _run_loss(model, params, obs_tokens, actions, rewards, ends, mask_padding):
    batch = {
        "obs_tokens": obs_tokens,
        "actions": actions,
        "rewards": rewards,
        "ends": ends,
        "mask_padding": mask_padding,
    }
    return model.apply(
        params,
        batch,
        jax.random.PRNGKey(1),
        method=model.compute_loss,
        rngs={"dropout": jax.random.PRNGKey(2)},
    )


def _expected_total_loss(
    *,
    obs_vocab_size: int,
    reward_num_classes: int,
    total_tokens: int,
    obs_ar_count: int,
    obs_btf_count: int,
    rew_count: int,
    end_count: int,
    mode: str,
    ar_aux_weight: float,
):
    obs_ar_sum = obs_ar_count * np.log(obs_vocab_size)
    obs_btf_sum = obs_btf_count * np.log(obs_vocab_size)
    rew_sum = rew_count * np.log(reward_num_classes)
    end_sum = end_count * np.log(2)
    if mode == "autoregressive":
        obs_term = obs_ar_sum
    elif mode == "block_teacher_forcing":
        obs_term = obs_btf_sum
    elif mode == "hybrid_btf_ar":
        obs_term = obs_btf_sum + ar_aux_weight * obs_ar_sum
    else:
        raise ValueError(mode)
    return (obs_term + rew_sum + end_sum) / total_tokens


def test_autoregressive_alignment_matches_shift_count():
    model, params = _build_model("autoregressive")

    # B=1, T=3, K=2
    obs_tokens = jnp.array([[[1, 2], [3, 4], [5, 6]]], dtype=jnp.int32)
    actions = jnp.array([[0, 1, 2]], dtype=jnp.int32)
    rewards = jnp.zeros((1, 3), dtype=jnp.float32)
    ends = jnp.zeros((1, 3), dtype=jnp.int32)
    mask_padding = jnp.array([[False, False, False]])

    out = _run_loss(model, params, obs_tokens, actions, rewards, ends, mask_padding)

    # AR label count with current shift alignment: B * (T*K - 1) = 5
    expected = _expected_total_loss(
        obs_vocab_size=7,
        reward_num_classes=2,
        total_tokens=1 * 3 * 3,
        obs_ar_count=5,
        obs_btf_count=4,
        rew_count=3,
        end_count=3,
        mode="autoregressive",
        ar_aux_weight=0.25,
    )
    assert np.isclose(float(out.total_loss), expected, rtol=1e-5, atol=1e-6)
    assert np.isclose(float(out.loss_obs_ar), np.log(7), rtol=1e-5, atol=1e-6)


def test_btf_uses_next_timestep_obs_and_drops_last_pair():
    model, params = _build_model("block_teacher_forcing")

    obs_tokens = jnp.array([[[1, 2], [3, 4], [5, 6]]], dtype=jnp.int32)
    actions = jnp.array([[0, 1, 2]], dtype=jnp.int32)
    rewards = jnp.zeros((1, 3), dtype=jnp.float32)
    ends = jnp.zeros((1, 3), dtype=jnp.int32)
    mask_padding = jnp.array([[False, False, False]])

    out = _run_loss(model, params, obs_tokens, actions, rewards, ends, mask_padding)

    # BTF valid obs pairs: B * (T-1) * K = 4 (t=0,1 only)
    expected = _expected_total_loss(
        obs_vocab_size=7,
        reward_num_classes=2,
        total_tokens=1 * 3 * 3,
        obs_ar_count=5,
        obs_btf_count=4,
        rew_count=3,
        end_count=3,
        mode="block_teacher_forcing",
        ar_aux_weight=0.25,
    )
    assert np.isclose(float(out.total_loss), expected, rtol=1e-5, atol=1e-6)
    assert np.isclose(float(out.loss_obs_btf), np.log(7), rtol=1e-5, atol=1e-6)


def test_btf_padding_requires_both_timesteps_valid():
    model, params = _build_model("block_teacher_forcing")

    # mask_padding=True means ignore.
    # valid steps: t0, t1, t3. BTF valid pair only (t0 -> t1).
    obs_tokens = jnp.array([[[1, 2], [3, 4], [5, 6], [0, 1]]], dtype=jnp.int32)
    actions = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
    rewards = jnp.zeros((1, 4), dtype=jnp.float32)
    ends = jnp.zeros((1, 4), dtype=jnp.int32)
    mask_padding = jnp.array([[False, False, True, False]])

    out = _run_loss(model, params, obs_tokens, actions, rewards, ends, mask_padding)

    expected = _expected_total_loss(
        obs_vocab_size=7,
        reward_num_classes=2,
        total_tokens=1 * 4 * 3,
        obs_ar_count=0,   # unused in pure BTF objective
        obs_btf_count=2,  # only one valid pair * K=2
        rew_count=3,
        end_count=3,
        mode="block_teacher_forcing",
        ar_aux_weight=0.25,
    )
    assert np.isclose(float(out.total_loss), expected, rtol=1e-5, atol=1e-6)


def test_hybrid_uses_weighted_btf_plus_ar_obs_terms():
    ar_aux_weight = 0.25
    model, params = _build_model("hybrid_btf_ar", ar_aux_weight=ar_aux_weight)

    obs_tokens = jnp.array([[[1, 2], [3, 4], [5, 6]]], dtype=jnp.int32)
    actions = jnp.array([[0, 1, 2]], dtype=jnp.int32)
    rewards = jnp.zeros((1, 3), dtype=jnp.float32)
    ends = jnp.zeros((1, 3), dtype=jnp.int32)
    mask_padding = jnp.array([[False, False, False]])

    out = _run_loss(model, params, obs_tokens, actions, rewards, ends, mask_padding)

    expected = _expected_total_loss(
        obs_vocab_size=7,
        reward_num_classes=2,
        total_tokens=1 * 3 * 3,
        obs_ar_count=5,
        obs_btf_count=4,
        rew_count=3,
        end_count=3,
        mode="hybrid_btf_ar",
        ar_aux_weight=ar_aux_weight,
    )
    assert np.isclose(float(out.total_loss), expected, rtol=1e-5, atol=1e-6)
