# world_model/algo5_train.py
# Algorithm 5 trainer with W&B logging, validation, PSNR, bits-per-token
import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import jax
import jax.numpy as jnp
import optax
import numpy as np

# --- project paths (same pattern you used) ---
BASE = Path(__file__).resolve().parent
sys.path.extend([
    str(BASE / "third_party" / "nanoGPT-jax"),
    str(BASE / "tokenizer"),
])

# Your modules
from vq_vae import VQVAE
from model import GPT, GPTConfig
from sample import make_replay_samplers  # your sampler factory over the Vault

# optional W&B
try:
    import wandb
except Exception:
    wandb = None


# ---------------- Utils ----------------
def tokens_to_quantized(tokens: jnp.ndarray, codebook: jnp.ndarray) -> jnp.ndarray:
    """
    tokens: (N, S) or (N, h, w) int32
    codebook: (V, E) float32   (E = embed_dim = 128)
    returns: (N, h, w, E) float32 suitable for VQ-VAE decoder
    """
    if tokens.ndim == 3:
        N, h, w = tokens.shape
        S = h * w
        flat = tokens.reshape(N, S)
        hh, ww = h, w
    elif tokens.ndim == 2:
        N, S = tokens.shape
        hh = ww = int(np.sqrt(S))
        assert hh * ww == S, f"S={S} is not a square grid"
        flat = tokens
    else:
        raise ValueError(f"Unexpected tokens shape {tokens.shape}")

    # Look up code vectors
    zq = codebook[flat]                  # (N, S, E)
    zq = zq.reshape(N, hh, ww, codebook.shape[1])  # (N, h, w, E)
    return zq

def bits_per_token(xent: float) -> float:
    # xent is in nats if computed via softmax_xent; convert to bits
    return float(xent / jnp.log(2.0))


def make_valid_mask(dones: jnp.ndarray) -> jnp.ndarray:
    """
    dones: (B, T) bool
    Returns mask over (B, T-1) that is 1 until (and excluding) the first done.
    """
    c = jnp.cumsum(dones.astype(jnp.int32), axis=1)
    valid = (c == 0)
    return valid[:, :-1]


def encode_obs_to_tokens(vq: VQVAE, params_vq, obs_f32: jnp.ndarray) -> jnp.ndarray:
    """
    obs_f32: (B, T, 63, 63, 3) in [0,1]
    returns tokens: (B, T, S) int32, where S = h*w from the encoder output
    """
    B, T, H, W, C = obs_f32.shape
    flat = obs_f32.reshape((-1, H, W, C))  # (BT, 63, 63, 3)
    _, _, _, _, tokens = vq.apply({'params': params_vq}, flat)  # usually (BT, h, w) int
    if tokens.ndim == 3:
        BT, h, w = tokens.shape
        tokens = tokens.reshape(B, T, h * w)
    elif tokens.ndim == 2:
        # already flattened: (BT, S)
        BT, S = tokens.shape
        tokens = tokens.reshape(B, T, S)
    else:
        raise ValueError(f"Unexpected tokens shape {tokens.shape}")
    return tokens.astype(jnp.int32)



def decode_tokens(vq: VQVAE, params_vq, tokens: jnp.ndarray) -> jnp.ndarray:
    """
    tokens: (B, S) int32
    decodes back to frames (B, 63, 63, 3) using your codebook + vq.decode(z_quantized).
    """
    codebook = params_vq['quantizer']['codebook']  # (V, D)
    B, S = tokens.shape
    # infer spatial grid (assume square)
    hw = int(round(S ** 0.5))
    assert hw * hw == S, f"Tokens length {S} is not a perfect square."
    D = codebook.shape[1]
    # codebook lookup -> (B, S, D) -> (B, h, w, D)
    z_q = codebook[tokens]                     # (B, S, D)
    z_q = z_q.reshape(B, hw, hw, D)           # spatial quantized
    frames = vq.apply({'params': params_vq}, z_q, method='decode')  # (B,63,63,3)
    return frames


def one_step_val_metrics(
    vq: VQVAE,
    params_vq,
    gpt: GPT,
    params_gpt,
    vb: dict,
    vocab_size: int,
    S: int,
) -> Tuple[float, float]:
    """
    Computes validation loss (nats/token) and PSNR for one-step prediction.
    """
    z = encode_obs_to_tokens(vq, params_vq, vb["obs"])  # (B,T,S)
    z_in, z_tgt = z[:, :-1, :], z[:, 1:, :]
    B, Tm1, S_ = z_in.shape
    assert S_ == S

    x_tok = z_in.reshape((-1, S))
    valid = make_valid_mask(vb["dones"]).reshape((-1, 1))  # (B*(T-1),1)

    logits, _ = gpt.apply({'params': params_gpt}, x_tok, train=False)  # (BTm1, S, vocab)
    ce = optax.softmax_cross_entropy_with_integer_labels(
        logits.reshape(-1, vocab_size),
        z_tgt.reshape(-1)
    ).reshape(B*Tm1, S)

    m = jnp.broadcast_to(valid, ce.shape).astype(jnp.float32)
    val_loss = (ce * m).sum() / jnp.maximum(1.0, m.sum())  # nats/token

    # greedy tokens -> decode next frame -> PSNR
    pred_tokens = jnp.argmax(logits, axis=-1).reshape(z_tgt.shape)  # (B,T-1,S)
    # take only the immediate next frame target at step 1 (teacher-forced context length 1)
    pred_next = decode_tokens(vq, params_vq, pred_tokens[:, 0, :])   # (B,63,63,3)
    gt_next = vb["obs"][:, 1]                                        # (B,63,63,3)
    mse = jnp.mean((pred_next - gt_next) ** 2)
    psnr = 10.0 * jnp.log10(1.0 / (mse + 1e-8))

    return float(val_loss), float(psnr)


# --------------- Main training (Algorithm 5) ---------------
def train_algo5(
    vault_uid: str,
    rel_dir: str = "/home/synaderi/Craftax_Baselines",
    T_wm: int = 20,
    batch_envs: int = 48,
    Nmb_WM: int = 3,
    N_iters_tok: int = 25,
    N_iters_TWM: int = 500,
    codebook_size: int = 512,
    vq_embed_dim: int = 128,
    gpt_layers: int = 8,
    gpt_heads: int = 8,
    gpt_embd: int = 512,
    tok_lr: float = 1e-3,
    twm_lr: float = 3e-4,
    weight_decay: float = 0.01,
    seed: int = 0,
    log_every: int = 10,
    val_every: int = 25,
    media_every: int = 100,
    wandb_project: str = "",
    wandb_entity: str = "",
    run_name: str = "",
):
    use_wandb = (wandb is not None) and (len(wandb_project) > 0)
    if use_wandb:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity or None,
            name=run_name or f"algo5_{vault_uid}",
            config=dict(
                vault_uid=vault_uid, rel_dir=rel_dir,
                T_wm=T_wm, batch_envs=batch_envs, Nmb_WM=Nmb_WM,
                N_iters_tok=N_iters_tok, N_iters_TWM=N_iters_TWM,
                codebook_size=codebook_size, vq_embed_dim=vq_embed_dim,
                gpt_layers=gpt_layers, gpt_heads=gpt_heads, gpt_embd=gpt_embd,
                tok_lr=tok_lr, twm_lr=twm_lr, weight_decay=weight_decay,
                seed=seed, val_every=val_every, media_every=media_every
            )
        )

    # 1) Samplers over the Vault
    sample_train, sample_val, info = make_replay_samplers(
        vault_uid=vault_uid,
        rel_dir=rel_dir,
        T_wm=T_wm,
        batch_envs=batch_envs,
    )
    print(f"[D] Loaded replay: B={info['B']}  T_train={info['T_train']}  T_val={info['T_val']}")

    rng = jax.random.PRNGKey(seed)

    # 2) Tokenizer (VQ-VAE)
    vq = VQVAE(vocab_size=codebook_size, embed_dim=vq_embed_dim)
    dummy = jnp.zeros((1, 63, 63, 3), jnp.float32)
    params_vq = vq.init(jax.random.PRNGKey(123), dummy)['params']

    opt_tok = optax.adam(tok_lr)
    opt_tok_state = opt_tok.init(params_vq)

    @jax.jit
    def tok_step(params, opt_state, batch_obs):
        x = batch_obs.reshape((-1, 63, 63, 3))  # train frame-wise
        def loss_fn(p):
            total, logs = vq.apply({'params': p}, x, method=vq.calculate_loss)
            # logs is a dict of DeviceArrays; leave as-is here
            return total, logs
        (loss, logs), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = opt_tok.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, logs
 

    print(f"[TOK] Training tokenizer for {N_iters_tok} iters × {Nmb_WM} minibatches")
    last_tok_loss = 0.0
    for it in range(N_iters_tok):
        for _ in range(Nmb_WM):
            rng, sub = jax.random.split(rng)
            batch = sample_train(sub)  # dict with obs/actions/rewards/dones
            params_vq, opt_tok_state, last_tok_loss, last_tok_logs = tok_step(params_vq, opt_tok_state, batch["obs"])
        if (it % log_every) == 0:
            print(f"[TOK] it={it:04d} loss={float(last_tok_loss):.4f}")
        if use_wandb:
            tok_log = {"tok/loss": float(last_tok_loss), "tok/iter": it}
            for k, v in last_tok_logs.items():
                tok_log[f"tok/{k}"] = float(v)
            wandb.log(tok_log)

    # 3) TWM (GPT) — determine S from a probe
    rng, sub = jax.random.split(rng)
    probe = sample_train(sub)["obs"]                           # (B,T,H,W,C)
    probe_tokens = encode_obs_to_tokens(vq, params_vq, probe)  # (B,T,S)
    S = int(probe_tokens.shape[-1])
    print(f"[TWM] spatial tokens S = {S}")

    gpt_cfg = GPTConfig(
        block_size=S, vocab_size=codebook_size,
        n_layer=gpt_layers, n_head=gpt_heads, n_embd=gpt_embd, dropout=0.0
    )
    gpt = GPT(gpt_cfg)
    params_gpt = gpt.init(
        {'params': jax.random.PRNGKey(321), 'dropout': jax.random.PRNGKey(321)},
        jnp.zeros((batch_envs, S), jnp.int32),
        train=True
    )['params']

    opt_twm = optax.adamw(twm_lr, weight_decay=weight_decay)
    opt_twm_state = opt_twm.init(params_gpt)

    @jax.jit
    def twm_step(params_gpt, opt_state, obs, dones, key):
        z = encode_obs_to_tokens(vq, params_vq, obs)     # (B,T,S)
        z_in, z_tgt = z[:, :-1, :], z[:, 1:, :]
        B, Tm1, S_ = z_in.shape
        x_tok = z_in.reshape((-1, S_))                    # (B*(T-1), S)
        valid = make_valid_mask(dones).reshape((-1, 1))   # (B*(T-1),1)

        def loss_fn(p):
            logits, _ = gpt.apply({'params': p}, x_tok, train=True, rngs={'dropout': key})
            ce = optax.softmax_cross_entropy_with_integer_labels(
                logits.reshape(-1, gpt_cfg.vocab_size),
                z_tgt.reshape(-1)
            ).reshape(B*Tm1, S_)
            m = jnp.broadcast_to(valid, ce.shape).astype(jnp.float32)
            loss = (ce * m).sum() / jnp.maximum(1.0, m.sum())
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(params_gpt)
        updates, opt_state = opt_twm.update(grads, opt_state, params_gpt)
        params_gpt = optax.apply_updates(params_gpt, updates)
        return params_gpt, opt_state, loss

    print(f"[TWM] Training GPT for {N_iters_TWM} iters × {Nmb_WM} minibatches (S={S})")
    last_twm_loss = 0.0
    for it in range(N_iters_TWM):
        for _ in range(Nmb_WM):
            rng, sub = jax.random.split(rng)
            batch = sample_train(sub)
            params_gpt, opt_twm_state, last_twm_loss = twm_step(
                params_gpt, opt_twm_state, batch["obs"], batch["dones"], sub
            )

        logs = {"twm/loss": float(last_twm_loss), "twm/iter": it, "twm/bpt": bits_per_token(float(last_twm_loss))}
        if (it % val_every) == 0:
            rng, sub = jax.random.split(rng)
            vb = sample_val(sub)
            val_loss, psnr = one_step_val_metrics(vq, params_vq, gpt, params_gpt, vb, gpt_cfg.vocab_size, S)
            logs.update({"twm/val_loss": val_loss, "twm/val_bpt": bits_per_token(val_loss), "twm/val_psnr": psnr})

        if (it % log_every) == 0:
            msg = f"[TWM] it={it:04d} loss={logs['twm/loss']:.4f}"
            if "twm/val_loss" in logs:
                msg += f"  val={logs['twm/val_loss']:.4f}  psnr={logs['twm/val_psnr']:.2f}dB"
            print(msg)

        if use_wandb:
            wandb.log(logs)

        # optional: tiny recon panel for visual sanity
        if use_wandb and (it % media_every) == 0:
            with jax.disable_jit():
                rng, sub = jax.random.split(rng)
                # batch for visualization
                batch = sample_train(sub) # or sample_val(sub) for consistency
                # Suppose you want to visualize/validate the *last* frame of the current train batch:
                obs_bt = batch["obs"]                              # (B, T, H, W, C)
                B_, T_ = obs_bt.shape[:2]

                # Encode to tokens (B,T,S) int32
                toks_bts = encode_obs_to_tokens(vq, params_vq, obs_bt)  # (B, T, S)
                S = toks_bts.shape[-1]
                h = w = int(np.sqrt(S)); assert h*w == S

                # Pick a slice to reconstruct, e.g. the last time step:
                toks_last = toks_bts[:, -1, :]                      # (B, S)

                # Map tokens -> code vectors using the tokenizer's codebook
                codebook = params_vq["quantizer"]["codebook"]       # (V, E=128)
                zq = tokens_to_quantized(toks_last, codebook)       # (B, h, w, 128)

                # Decode
                rec = vq.apply({'params': params_vq}, zq, method='decode')  # (B, 63, 63, 3)
                rec = jnp.clip(rec, 0.0, 1.0)

                # Compute PSNR vs ground truth for that frame:
                gt = obs_bt[:, -1, ...]                             # (B, 63, 63, 3)
                mse = jnp.mean((rec - gt) ** 2)
                psnr = 10.0 * jnp.log10(1.0 / jnp.maximum(mse, 1e-8))

                imgs = np.array(gt[:8])  # 8 real frames at t=0
                rec = np.array(rec[:8])
                # side-by-side strip per image
                panel = [wandb.Image((np.concatenate([imgs[i], rec[i]], axis=1) * 255).astype(np.uint8),
                                     caption=f"real | recon #{i}  PSNR={psnr:.2f}dB") for i in range(len(imgs))]
                wandb.log({"media/recon_panel": panel, "twm/iter": it, "media/psnr": float(psnr)})

    print("\n✅ Algo 5 finished.")
    if use_wandb:
        wandb.finish()

    return (vq, params_vq), (gpt, params_gpt)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vault_uid", required=True)
    ap.add_argument("--rel_dir", default="/home/synaderi/Craftax_Baselines")
    ap.add_argument("--T_wm", type=int, default=20)
    ap.add_argument("--batch_envs", type=int, default=48)
    ap.add_argument("--Nmb_WM", type=int, default=3)
    ap.add_argument("--N_iters_tok", type=int, default=25)      # paper: 500 (M1-3)
    ap.add_argument("--N_iters_TWM", type=int, default=500)     # paper: 500
    ap.add_argument("--codebook_size", type=int, default=512)
    ap.add_argument("--vq_embed_dim", type=int, default=128)
    ap.add_argument("--gpt_layers", type=int, default=8)
    ap.add_argument("--gpt_heads", type=int, default=8)
    ap.add_argument("--gpt_embd", type=int, default=512)
    ap.add_argument("--tok_lr", type=float, default=1e-3)
    ap.add_argument("--twm_lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--val_every", type=int, default=25)
    ap.add_argument("--media_every", type=int, default=100)
    ap.add_argument("--wandb_project", type=str, default="")
    ap.add_argument("--wandb_entity", type=str, default="")
    ap.add_argument("--run_name", type=str, default="")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_algo5(
        vault_uid=args.vault_uid,
        rel_dir=args.rel_dir,
        T_wm=args.T_wm,
        batch_envs=args.batch_envs,
        Nmb_WM=args.Nmb_WM,
        N_iters_tok=args.N_iters_tok,
        N_iters_TWM=args.N_iters_TWM,
        codebook_size=args.codebook_size,
        vq_embed_dim=args.vq_embed_dim,
        gpt_layers=args.gpt_layers,
        gpt_heads=args.gpt_heads,
        gpt_embd=args.gpt_embd,
        tok_lr=args.tok_lr,
        twm_lr=args.twm_lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        log_every=args.log_every,
        val_every=args.val_every,
        media_every=args.media_every,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        run_name=args.run_name,
    )
