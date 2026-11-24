"""JAX interpreter for validated programs, traced once per genotype."""
import jax
import jax.numpy as jnp


def execute(graph, inputs):
    values = dict(inputs)
    for node in graph['nodes']:
        op = node['op']
        if op == 'constant':
            result = jnp.asarray(node['value'], dtype=jnp.float32)
        else:
            args = [values[a] for a in node['args']]
            a = args[0]
            if op == 'add': result = a + args[1]
            elif op == 'sub': result = a - args[1]
            elif op == 'mul': result = a * args[1]
            elif op == 'div': result = a / jnp.where(jnp.abs(args[1]) < 1e-8, jnp.where(args[1] < 0, -1e-8, 1e-8), args[1])
            elif op == 'minimum': result = jnp.minimum(a, args[1])
            elif op == 'maximum': result = jnp.maximum(a, args[1])
            elif op == 'neg': result = -a
            elif op == 'square': result = jnp.square(a)
            elif op == 'abs': result = jnp.abs(a)
            elif op == 'tanh': result = jnp.tanh(a)
            elif op == 'exp': result = jnp.exp(a)
            elif op == 'logabs': result = jnp.log(jnp.abs(a) + 1e-8)
            elif op == 'mean': result = jnp.mean(a)
            elif op == 'std': result = jnp.std(a)
            elif op == 'detach': result = jax.lax.stop_gradient(a)
            elif op == 'clip': result = jnp.clip(a, args[1], args[2])
            else: raise ValueError(op)
        values[node['id']] = result
    return tuple(values[out] for out in graph['outputs'])


def common(memory, gamma, lam, progress):
    return dict(gamma=gamma, lam=lam, progress=progress, **{f'm{i}':memory[i] for i in range(4)})


def credit_targets(program, reward, value, done, last_value, memory, gamma, lam, progress):
    shared = common(memory, gamma, lam, progress)
    def step(carry_and_value, transition):
        carry, next_value = carry_and_value
        r, v, terminal = transition
        live = 1. - terminal.astype(jnp.float32)
        # These masks enforce episode boundaries for EVERY evolved recurrence.
        inputs = dict(shared, reward=r, value=v, next_value=next_value * live,
                      live=live, carry=carry * live)
        credit = jnp.broadcast_to(execute(program['credit'], inputs)[0], v.shape)
        return (credit, v), credit
    _, credit = jax.lax.scan(step, (jnp.zeros_like(last_value), last_value),
                            (reward, value, done), reverse=True, unroll=16)
    target = execute(program['target'], dict(shared, reward=reward, value=value, credit=credit))[0]
    return jax.lax.stop_gradient(credit), jax.lax.stop_gradient(jnp.broadcast_to(target, value.shape))


def program_loss(program, logp, old_logp, value, old_value, advantage, target,
                 entropy, memory, gamma, lam, progress, clip_eps, vf_coef, ent_coef):
    return execute(program['loss'], dict(common(memory, gamma, lam, progress),
        logp=logp, old_logp=old_logp, value=value, old_value=old_value,
        advantage=advantage, target=target, entropy=entropy,
        clip_eps=clip_eps, vf_coef=vf_coef, ent_coef=ent_coef))[0]


def update_memory(program, memory, reward, value, done, credit, gamma, lam, progress):
    inputs = dict(common(memory, gamma, lam, progress), reward_mean=reward.mean(),
        reward_std=reward.std(), value_mean=value.mean(), credit_mean=credit.mean(),
        credit_std=credit.std(), done_mean=done.astype(jnp.float32).mean())
    return jax.lax.stop_gradient(jnp.stack(execute(program['memory'], inputs)))
