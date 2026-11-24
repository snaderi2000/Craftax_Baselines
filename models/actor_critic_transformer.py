"""Exact rolling-observation causal Transformer for symbolic recurrent PPO.

Each output uses at most context-1 previous observations plus the current one.
Raw observations (not stale learned embeddings) cross rollout boundaries.
Independent windows ensure two attention layers cannot extend the receptive
field beyond the requested context. Rematerialized scanning bounds activation
memory while keeping online and sequence evaluation equivalent.
"""
import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import orthogonal,constant


class WindowTransformer(nn.Module):
    width:int
    heads:int
    layers:int
    context:int

    @nn.compact
    def __call__(self,carry,inputs):
        tokens,valid=inputs  # B,C,E; B,C
        position=self.param('position',nn.initializers.normal(.02),(self.context,self.width))
        x=tokens+position[None]
        causal=jnp.tril(jnp.ones((self.context,self.context),bool))
        mask=causal[None,None]&valid[:,None,None,:]
        for layer in range(self.layers):
            y=nn.LayerNorm(name=f'attn_norm_{layer}')(x)
            y=nn.MultiHeadDotProductAttention(num_heads=self.heads,qkv_features=self.width,
                out_features=self.width,dropout_rate=0.,name=f'attn_{layer}')(y,y,mask=mask,deterministic=True)
            x=x+y
            y=nn.LayerNorm(name=f'ff_norm_{layer}')(x)
            y=nn.gelu(nn.Dense(4*self.width,name=f'ff_in_{layer}')(y))
            x=x+nn.Dense(self.width,name=f'ff_out_{layer}')(y)
        return carry,nn.LayerNorm(name='output_norm')(x[:,-1])


class ActorCriticTransformer(nn.Module):
    action_dim:int
    config:dict

    @staticmethod
    def initialize_carry(batch_size,context=64,observation_dim=1345):
        return (jnp.zeros((batch_size,context-1,observation_dim),jnp.float32),
                jnp.zeros((batch_size,context-1),bool))

    @nn.compact
    def __call__(self,carry,inputs):
        obs,resets=inputs  # T,B,D; T,B
        context=int(self.config.get('TRANSFORMER_CONTEXT',64))
        width=int(self.config.get('TRANSFORMER_DIM',256))
        heads=int(self.config.get('TRANSFORMER_HEADS',4))
        layers=int(self.config.get('TRANSFORMER_LAYERS',2))
        history,history_valid=carry
        sequence=jnp.concatenate((history,jnp.swapaxes(obs,0,1)),axis=1)
        valid=jnp.concatenate((history_valid,jnp.ones(obs.shape[:2][::-1],bool)),axis=1)
        segments=jnp.concatenate((jnp.zeros_like(history_valid,dtype=jnp.int32),
            jnp.cumsum(jnp.swapaxes(resets,0,1).astype(jnp.int32),axis=1)),axis=1)
        # Every causal window is recomputed under the CURRENT parameters.
        embedding=nn.relu(nn.Dense(width,kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0),name='encoder')(sequence))
        indices=jnp.arange(obs.shape[0])[:,None]+jnp.arange(context)[None,:]
        windows=jnp.swapaxes(jnp.take(embedding,indices,axis=1),0,1)
        window_segments=jnp.take(segments,indices,axis=1)
        window_valid=jnp.take(valid,indices,axis=1)&(window_segments==window_segments[:,:,-1:])
        scan=nn.scan(nn.remat(WindowTransformer),variable_broadcast='params',
            split_rngs={'params':False},in_axes=0,out_axes=0)
        _,features=scan(width,heads,layers,context,name='temporal')(
            jnp.asarray(0),(windows,jnp.swapaxes(window_valid,0,1)))
        next_valid=valid[:,-(context-1):]&(segments[:,-(context-1):]==segments[:,-1:])
        next_carry=(sequence[:,-(context-1):],next_valid)
        size=int(self.config.get('LAYER_SIZE',512))
        actor=features;critic=features
        for i in range(2):
            actor=nn.relu(nn.Dense(size,kernel_init=orthogonal(2),bias_init=constant(0),name=f'actor_{i}')(actor))
            critic=nn.relu(nn.Dense(size,kernel_init=orthogonal(2),bias_init=constant(0),name=f'critic_{i}')(critic))
        logits=nn.Dense(self.action_dim,kernel_init=orthogonal(.01),bias_init=constant(0),name='actor_out')(actor)
        values=nn.Dense(1,kernel_init=orthogonal(1),bias_init=constant(0),name='critic_out')(critic)
        return next_carry,distrax.Categorical(logits=logits),jnp.squeeze(values,-1)
