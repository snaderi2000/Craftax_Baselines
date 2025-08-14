import sys
import pathlib
import jax
import jax.numpy as jnp
import optax

# ---- Add submodule to path ----
SUBMODULE = pathlib.Path(__file__).resolve().parents[1] / "third_party" / "nanoGPT-jax"
sys.path.append(str(SUBMODULE))

from model import GPTConfig, GPT  # noqa: E402


def cross_entropy_with_mask(logits, targets, mask):
    logp = jax.nn.log_softmax(logits, axis=-1)
    ll = jnp.take_along_axis(logp, targets[..., None], axis=-1).squeeze(-1)
    return -(ll * mask).sum() / (mask.sum() + 1e-8)


def test_forward_loss_and_grads():
    vocab_size, seq_len = 520, 10
    conf = GPTConfig(vocab_size=vocab_size, block_size=seq_len, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    model = GPT(conf)

    B = 4
    x = jnp.ones((B, seq_len), dtype=jnp.int32)
    params = model.init(jax.random.PRNGKey(0), x, train=False)

    logits, _ = model.apply(params, x, train=False)
    assert logits.shape == (B, seq_len, vocab_size)

    targets = jnp.ones((B, seq_len), dtype=jnp.int32)
    mask = jnp.zeros((B, seq_len), dtype=jnp.float32).at[:, -2:].set(1.0)
    loss = cross_entropy_with_mask(logits, targets, mask)
    assert jnp.isfinite(loss)

    def loss_fn(p):
        l, _ = model.apply(p, x, train=False)
        return cross_entropy_with_mask(l, targets, mask)

    grads = jax.grad(loss_fn)(params)
    leaves = jax.tree_util.tree_leaves(grads)
    assert leaves and all(jnp.all(jnp.isfinite(g)) for g in leaves)


def test_causal_mask_no_future_leak():
    conf = GPTConfig(vocab_size=100, block_size=8, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    model = GPT(conf)
    B, T = 2, 8
    x = jax.random.randint(jax.random.PRNGKey(0), (B, T), 0, conf.vocab_size)
    params = model.init(jax.random.PRNGKey(1), x, train=False)

    logits_a, _ = model.apply(params, x, train=False)

    # Change ONLY the last token; earlier logits should not change
    x2 = x.at[:, -1].set((x[:, -1] + 1) % conf.vocab_size)
    logits_b, _ = model.apply(params, x2, train=False)

    assert jnp.allclose(logits_a[:, :-1, :], logits_b[:, :-1, :], atol=1e-6)


def test_jit_forward():
    conf = GPTConfig(vocab_size=128, block_size=12, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    model = GPT(conf)
    x = jnp.ones((4, 12), dtype=jnp.int32)
    params = model.init(jax.random.PRNGKey(0), x, train=False)

    @jax.jit
    def f(p, inp):
        return model.apply(p, inp, train=False)[0]

    logits = f(params, x)
    assert logits.shape == (4, 12, 128)
    assert jnp.isfinite(logits).all()


def test_dropout_rng_changes_output():
    conf = GPTConfig(vocab_size=64, block_size=6, n_layer=2, n_head=2, n_embd=32, dropout=0.1)
    model = GPT(conf)
    x = jnp.ones((2, 6), dtype=jnp.int32)

    # init with train=True so dropout shapes are set
    params = model.init(jax.random.PRNGKey(0), x, train=True)

    rngs_a = {"dropout": jax.random.PRNGKey(42)}
    rngs_b = {"dropout": jax.random.PRNGKey(43)}

    logits1, _ = model.apply(params, x, train=True, rngs=rngs_a)
    logits2, _ = model.apply(params, x, train=True, rngs=rngs_a)
    logits3, _ = model.apply(params, x, train=True, rngs=rngs_b)

    assert jnp.allclose(logits1, logits2)      # same key -> same output
    assert not jnp.allclose(logits1, logits3)  # different key -> (very likely) different


def test_one_train_step_updates_params():
    conf = GPTConfig(vocab_size=128, block_size=8, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    model = GPT(conf)
    x = jax.random.randint(jax.random.PRNGKey(0), (4, 8), 0, conf.vocab_size)
    y = jax.random.randint(jax.random.PRNGKey(1), (4, 8), 0, conf.vocab_size)

    params = model.init(jax.random.PRNGKey(2), x, train=True)["params"]
    tx = optax.adam(1e-3)
    opt_state = tx.init(params)

    def loss_fn(p):
        logits, _ = model.apply({"params": p}, x, train=True, rngs={"dropout": jax.random.PRNGKey(0)})
        logp = jax.nn.log_softmax(logits, axis=-1)
        ll = jnp.take_along_axis(logp, y[..., None], axis=-1).squeeze(-1)
        return -ll.mean()

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    assert jnp.isfinite(loss)
    old_leaves = jax.tree_util.tree_leaves(params)
    new_leaves = jax.tree_util.tree_leaves(new_params)
    assert any(not jnp.allclose(a, b) for a, b in zip(old_leaves, new_leaves))


def test_generate_shape_and_range():
    conf = GPTConfig(vocab_size=256, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    model = GPT(conf)
    x = jax.random.randint(jax.random.PRNGKey(0), (2, 5), 0, conf.vocab_size)
    # params = model.init(jax.random.PRNGKey(1), x, train=False)

    # out = model.generate(jax.random.PRNGKey(2), params, x, max_new_tokens=7, temperature=1.0, top_k=50)
    variables = model.init(jax.random.PRNGKey(1), x, train=False)
    params = variables["params"]  # <-- extract raw params
    out = model.generate(jax.random.PRNGKey(2), params, x, max_new_tokens=7, temperature=1.0, top_k=50)

    assert out.shape == (2, 12)  # 5 + 7
    assert (out >= 0).all() and (out < conf.vocab_size).all()


def test_crop_block_size():
    conf = GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    model = GPT(conf)
    x = jnp.ones((1, 8), dtype=jnp.int32)
    params = model.init(jax.random.PRNGKey(0), x, train=False)["params"]

    cropped = model.crop_block_size(params, block_size=8)
    logits, _ = model.apply({"params": cropped}, x, train=False)
    assert logits.shape == (1, 8, 64)
