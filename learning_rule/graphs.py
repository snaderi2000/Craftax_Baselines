"""Typed, acyclic arithmetic programs. JSON only: no Python eval or execution.

S denotes a scalar and V an elementwise batch/trajectory vector. Scalars may
broadcast into V operations. Credit is a backwards recurrence whose carry is
reset at episode boundaries by the runtime. All four memory outputs are scalar.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math

ARITY = dict(add=2, sub=2, mul=2, div=2, minimum=2, maximum=2,
             neg=1, square=1, abs=1, tanh=1, exp=1, logabs=1,
             mean=1, std=1, detach=1, clip=3)
MEMORY = {f'm{i}':'S' for i in range(4)}
COMMON = dict(MEMORY, gamma='S', lam='S', progress='S')
INPUTS = dict(
    credit=dict(COMMON, reward='V', value='V', next_value='V', live='V', carry='V'),
    target=dict(COMMON, reward='V', value='V', credit='V'),
    loss=dict(COMMON, logp='V', old_logp='V', value='V', old_value='V',
              advantage='V', target='V', entropy='V', clip_eps='S', vf_coef='S', ent_coef='S'),
    memory=dict(COMMON, reward_mean='S', reward_std='S', value_mean='S',
                credit_mean='S', credit_std='S', done_mean='S'))


def graph_types(graph, inputs):
    if set(graph) != {'nodes', 'outputs'} or not 0 <= len(graph['nodes']) <= 128:
        raise ValueError('A graph needs 0..128 nodes and explicit outputs')
    types = dict(inputs)
    for node in graph['nodes']:
        name, op = node.get('id'), node.get('op')
        if not isinstance(name, str) or name in types:
            raise ValueError('Duplicate/invalid node ID')
        if op == 'constant':
            if set(node) != {'id', 'op', 'value'} or type(node['value']) not in (int, float) or not math.isfinite(node['value']) or abs(node['value']) > 100:
                raise ValueError('Invalid bounded constant')
            types[name] = 'S'
            continue
        if op not in ARITY or set(node) != {'id', 'op', 'args'} or len(node['args']) != ARITY[op]:
            raise ValueError('Unknown operation/arity')
        if any(arg not in types for arg in node['args']):
            raise ValueError('Unknown input, cycle, or forward reference')
        args = [types[a] for a in node['args']]
        if op == 'clip' and args[1:] != ['S', 'S']:
            raise ValueError('Clip bounds must be scalar')
        types[name] = 'S' if op in ('mean', 'std') else ('V' if 'V' in args else 'S')
    if not graph['outputs'] or any(out not in types for out in graph['outputs']):
        raise ValueError('Missing graph output')
    return types


def prune(graph):
    """Remove unreachable nodes so inert mutations have identical fingerprints."""
    graph = deepcopy(graph)
    by_id = {n['id']:n for n in graph['nodes']}
    used = set()
    def visit(name):
        if name in used or name not in by_id:
            return
        used.add(name)
        for arg in by_id[name].get('args', []):
            visit(arg)
    for out in graph['outputs']:
        visit(out)
    graph['nodes'] = [n for n in graph['nodes'] if n['id'] in used]
    return graph


def canonical_graph(graph):
    graph = prune(graph)
    mapping = {n['id']:f'n{i}' for i, n in enumerate(graph['nodes'])}
    for node in graph['nodes']:
        node['id'] = mapping[node['id']]
        if 'args' in node:
            node['args'] = [mapping.get(a, a) for a in node['args']]
    graph['outputs'] = [mapping.get(a, a) for a in graph['outputs']]
    return graph


@dataclass(frozen=True)
class Program:
    """Canonical serialized genotype; trainable network weights are separate."""
    encoded: str

    @classmethod
    def from_dict(cls, data):
        if set(data) != {'schema', 'credit', 'target', 'loss', 'memory'} or data['schema'] != 1:
            raise ValueError('Unknown program schema')
        result = dict(schema=1)
        for stage, inputs in INPUTS.items():
            types = graph_types(data[stage], inputs)
            outputs = data[stage]['outputs']
            if len(outputs) != (4 if stage == 'memory' else 1):
                raise ValueError('Incorrect number of outputs')
            if stage in ('loss', 'memory') and any(types[o] != 'S' for o in outputs):
                raise ValueError('Loss and learning-memory outputs must be scalar')
            result[stage] = canonical_graph(data[stage])
        return cls(json.dumps(result, sort_keys=True, separators=(',', ':')))

    @property
    def id(self):
        return hashlib.sha256(self.encoded.encode()).hexdigest()[:12]

    def to_dict(self):
        return json.loads(self.encoded)


class Builder:
    def __init__(self):
        self.nodes = []

    def op(self, op, *args):
        name = f'x{len(self.nodes)}'
        self.nodes.append(dict(id=name, op=op, args=list(args)))
        return name

    def const(self, value):
        name = f'x{len(self.nodes)}'
        self.nodes.append(dict(id=name, op='constant', value=value))
        return name

    def finish(self, *outputs):
        return dict(nodes=self.nodes, outputs=list(outputs))


def ppo_program():
    c = Builder()
    boot = c.op('mul', c.op('mul', 'gamma', 'next_value'), 'live')
    delta = c.op('sub', c.op('add', 'reward', boot), 'value')
    continuation = c.op('mul', c.op('mul', c.op('mul', 'gamma', 'lam'), 'live'), 'carry')
    credit = c.finish(c.op('add', delta, continuation))
    t = Builder(); target = t.finish(t.op('add', 'credit', 'value'))
    l = Builder()
    one, half, eps = l.const(1.), l.const(.5), l.const(1e-8)
    ratio = l.op('exp', l.op('sub', 'logp', 'old_logp'))
    normalized = l.op('div', l.op('sub', 'advantage', l.op('mean', 'advantage')),
                      l.op('add', l.op('std', 'advantage'), eps))
    clipped = l.op('clip', ratio, l.op('sub', one, 'clip_eps'), l.op('add', one, 'clip_eps'))
    actor = l.op('mean', l.op('neg', l.op('minimum', l.op('mul', ratio, normalized), l.op('mul', clipped, normalized))))
    value_clip = l.op('add', 'old_value', l.op('clip', l.op('sub', 'value', 'old_value'), l.op('neg', 'clip_eps'), 'clip_eps'))
    critic = l.op('mul', half, l.op('mean', l.op('maximum',
        l.op('square', l.op('sub', 'value', 'target')),
        l.op('square', l.op('sub', value_clip, 'target')))))
    total = l.op('sub', l.op('add', actor, l.op('mul', 'vf_coef', critic)), l.op('mul', 'ent_coef', l.op('mean', 'entropy')))
    m = Builder(); zero = m.const(0.)
    return Program.from_dict(dict(schema=1, credit=credit, target=target,
                                 loss=l.finish(total), memory=m.finish(zero, zero, zero, zero)))


PPO = ppo_program()


def actor_critic_program():
    """A second known starting point, using log-policy loss without PPO clipping."""
    data = PPO.to_dict()
    b = Builder(); eps = b.const(1e-8); half = b.const(.5)
    normalized = b.op('div', b.op('sub', 'advantage', b.op('mean', 'advantage')),
                       b.op('add', b.op('std', 'advantage'), eps))
    actor = b.op('neg', b.op('mean', b.op('mul', 'logp', normalized)))
    critic = b.op('mul', half, b.op('mean', b.op('square', b.op('sub', 'value', 'target'))))
    total = b.op('sub', b.op('add', actor, b.op('mul', 'vf_coef', critic)),
                 b.op('mul', 'ent_coef', b.op('mean', 'entropy')))
    data['loss'] = b.finish(total)
    return Program.from_dict(data)


def describe(program):
    """Readable, exact program equations; no semantic novelty claims."""
    lines = [f'Program {program.id}']
    for stage, graph in program.to_dict().items():
        if stage == 'schema':
            continue
        lines.append(f'\n{stage}:')
        for node in graph['nodes']:
            rhs = repr(node['value']) if node['op'] == 'constant' else f"{node['op']}({', '.join(node['args'])})"
            lines.append(f"  {node['id']} = {rhs}")
        lines.append('  return ' + ', '.join(graph['outputs']))
    return '\n'.join(lines) + '\n'
