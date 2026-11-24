"""Structural graph mutation and a numerical screen, independent of fitness."""
from copy import deepcopy
import hashlib
import json
import math

from learning_rule.graphs import ARITY, INPUTS, Program, graph_types


def mutate(parent, rng):
    for _ in range(500):
        data = parent.to_dict()
        stage = rng.choices(['credit', 'target', 'loss', 'memory'], [3, 2, 5, 2])[0]
        graph = data[stage]
        nodes = graph['nodes']
        kinds = ['insert', 'rewire'] + (['replace', 'delete'] if nodes else [])
        kind = rng.choice(kinds)
        before = deepcopy(graph)
        if kind == 'insert':
            index = rng.randrange(len(nodes) + 1)
            prefix = dict(nodes=nodes[:index], outputs=list(INPUTS[stage])[:1])
            types = graph_types(prefix, INPUTS[stage])
            # Insert on an existing edge, then redirect that edge through the new node.
            sites = [(n, k, a) for n in nodes[index:] for k, a in enumerate(n.get('args', [])) if a in types]
            sites += [(graph, k, a) for k, a in enumerate(graph['outputs']) if a in types]
            if not sites: continue
            consumer, slot, old = rng.choice(sites)
            new_id = f'insert{len(nodes)}'
            while new_id in types or any(n['id'] == new_id for n in nodes): new_id += 'x'
            operation = rng.choice(list(ARITY))
            args = [old]
            for _ in range(ARITY[operation] - 1):
                args.append(rng.choice([n for n, typ in types.items() if operation != 'clip' or typ == 'S']))
            node = dict(id=new_id, op=operation, args=args)
            if consumer is graph: graph['outputs'][slot] = new_id
            else: consumer['args'][slot] = new_id
            nodes.insert(index, node)
        elif kind == 'rewire':
            index = rng.randrange(len(nodes) + 1)
            available = list(INPUTS[stage]) + [n['id'] for n in nodes[:index]]
            if index == len(nodes):
                slot = rng.randrange(len(graph['outputs']))
                graph['outputs'][slot] = rng.choice(available)
            elif nodes[index]['op'] != 'constant':
                slot = rng.randrange(len(nodes[index]['args']))
                nodes[index]['args'][slot] = rng.choice(available)
            else: continue
        elif kind == 'replace':
            node = rng.choice(nodes)
            if node['op'] == 'constant':
                node['value'] = round(rng.uniform(-2, 2), 6) if rng.random() < .4 else node['value'] * rng.choice([.5, 2., -1.])
            else:
                node['op'] = rng.choice([op for op in ARITY if ARITY[op] == len(node['args'])])
        else:
            index = rng.randrange(len(nodes)); removed = nodes[index]
            replacement = rng.choice(list(INPUTS[stage]) + [n['id'] for n in nodes[:index]])
            for node in nodes[index + 1:]:
                if 'args' in node: node['args'] = [replacement if a == removed['id'] else a for a in node['args']]
            graph['outputs'] = [replacement if a == removed['id'] else a for a in graph['outputs']]
            del nodes[index]
        try:
            child = Program.from_dict(data)
        except ValueError:
            continue
        if child.id != parent.id:
            return child, dict(stage=stage, operator=kind, before=before, after=graph,
                               parent=parent.id, child=child.id)
    raise RuntimeError('No valid structural mutation after 500 attempts')


def numerical_screen(program):
    """Synthetic data only; catches constants, nonfinite outputs and gradients.

    This is not an environment fitness estimate. Signatures include losses AND
    derivatives to distinguish stop-gradient edits with identical forward values.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from learning_rule.runtime import credit_targets, program_loss, update_memory
    data = program.to_dict()
    signatures = []
    memory = jnp.zeros(4)
    for phase in (0., .5, 1.):
        x = jnp.arange(24, dtype=jnp.float32).reshape(6, 4)
        reward = ((x % 7) == 0).astype(jnp.float32) - .1 * ((x % 11) == 0)
        value = jnp.sin(x * .2) + .1
        done = ((x % 13) == 0)
        credit, target = credit_targets(data, reward, value, done, value[-1], memory, .99, .8, phase)
        updated = update_memory(data, memory, reward, value, done, credit, .99, .8, phase)
        old_logp = -2.8 + .2 * jnp.cos(x.reshape(-1))
        def loss_fn(logp, v, ent):
            return program_loss(data, logp, old_logp, v, value.reshape(-1), credit.reshape(-1),
                target.reshape(-1), ent, memory, .99, .8, phase, .2, .5, .01)
        loss, gradients = jax.value_and_grad(loss_fn, argnums=(0, 1, 2))(
            old_logp + .03 * jnp.sin(x.reshape(-1)), value.reshape(-1) + .02, jnp.full(24, 2.7))
        arrays = [np.asarray(a) for a in (credit, target, updated, loss, *gradients)]
        if any(not np.all(np.isfinite(a)) or np.max(np.abs(a)) > 1e6 for a in arrays):
            return dict(valid=False, reason='Nonfinite or excessive synthetic output/gradient')
        if sum(float(np.sum(g * g)) for g in arrays[-3:]) < 1e-16:
            return dict(valid=False, reason='Loss has no effective network gradient')
        signatures.extend(np.round(a.astype(np.float64), 5).tolist() for i, a in enumerate(arrays) if i != 2)
        memory = updated
    signature = hashlib.sha256(json.dumps(signatures, sort_keys=True).encode()).hexdigest()
    return dict(valid=True, signature=signature)


def offspring(parent, rng, seen_ids, seen_signatures, edits=1, attempts=250):
    for _ in range(attempts):
        child, changes = parent, []
        for _ in range(edits):
            child, change = mutate(child, rng); changes.append(change)
        if child.id in seen_ids: continue
        screen = numerical_screen(child)
        if screen['valid'] and screen['signature'] not in seen_signatures:
            seen_ids.add(child.id); seen_signatures.add(screen['signature'])
            return child, changes, screen
    raise RuntimeError('Could not generate a distinct numerically valid program')
