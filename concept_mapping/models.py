from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp


def l2_normalize(x: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), eps)


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    out_dim: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        act = nn.relu if self.activation == "relu" else nn.tanh
        for width in self.hidden_dims:
            x = nn.Dense(width)(x)
            x = act(x)
        return nn.Dense(self.out_dim)(x)


class StateEncoder(nn.Module):
    latent_dim: int = 128
    hidden_dims: Sequence[int] = (512, 512)

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        obs = obs.astype(jnp.float32)
        z = MLP(self.hidden_dims, self.latent_dim)(obs)
        return nn.LayerNorm()(z)


class TransitionConceptEncoder(nn.Module):
    num_actions: int
    state_latent_dim: int = 128
    action_dim: int = 32
    concept_dim: int = 128
    hidden_dims: Sequence[int] = (512, 512)

    def setup(self) -> None:
        self.state_encoder = StateEncoder(
            latent_dim=self.state_latent_dim,
            hidden_dims=self.hidden_dims,
        )
        self.action_embedding = nn.Embed(
            num_embeddings=self.num_actions,
            features=self.action_dim,
        )
        self.psi_head = MLP(self.hidden_dims, self.concept_dim)
        self.nu_head = MLP(self.hidden_dims, self.concept_dim)
        self.forward_head = MLP(self.hidden_dims, self.state_latent_dim)

    def encode_state(self, obs: jnp.ndarray) -> jnp.ndarray:
        return self.state_encoder(obs)

    def encode_state_action(self, obs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        z = self.encode_state(obs)
        a = self.action_embedding(action.astype(jnp.int32))
        return l2_normalize(self.psi_head(jnp.concatenate([z, a], axis=-1)))

    def encode_transition(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        next_obs: jnp.ndarray,
    ) -> jnp.ndarray:
        z = self.encode_state(obs)
        z_next = self.encode_state(next_obs)
        a = self.action_embedding(action.astype(jnp.int32))
        features = jnp.concatenate([z, a, z_next - z], axis=-1)
        return l2_normalize(self.nu_head(features))

    def predict_next_latent(self, obs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        z = self.encode_state(obs)
        a = self.action_embedding(action.astype(jnp.int32))
        delta = self.forward_head(jnp.concatenate([z, a], axis=-1))
        return z + delta

    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        next_obs: jnp.ndarray,
    ) -> dict[str, jnp.ndarray]:
        z = self.encode_state(obs)
        z_next = self.encode_state(next_obs)
        a = self.action_embedding(action.astype(jnp.int32))
        psi = l2_normalize(self.psi_head(jnp.concatenate([z, a], axis=-1)))
        nu = l2_normalize(self.nu_head(jnp.concatenate([z, a, z_next - z], axis=-1)))
        pred_next_z = self.forward_head(jnp.concatenate([z, a], axis=-1)) + z
        return {
            "state": z,
            "next_state": z_next,
            "psi": psi,
            "nu": nu,
            "pred_next_state": pred_next_z,
        }


class ICMTransitionConceptEncoder(nn.Module):
    num_actions: int
    state_latent_dim: int = 128
    concept_dim: int = 128
    hidden_dims: Sequence[int] = (512, 512)

    def setup(self) -> None:
        self.state_encoder = StateEncoder(
            latent_dim=self.state_latent_dim,
            hidden_dims=self.hidden_dims,
        )
        self.inverse_head = MLP(self.hidden_dims, self.num_actions)
        self.forward_head = MLP(self.hidden_dims, self.state_latent_dim)
        self.concept_head = MLP(self.hidden_dims, self.concept_dim)
        self.reward_head = MLP((256,), 1)
        self.done_head = MLP((256,), 1)
        self.delta_head = MLP((256,), 16)

    def encode_state(self, obs: jnp.ndarray) -> jnp.ndarray:
        return self.state_encoder(obs)

    def encode_transition(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        next_obs: jnp.ndarray,
    ) -> jnp.ndarray:
        z = self.encode_state(obs)
        z_next = self.encode_state(next_obs)
        action_oh = jax.nn.one_hot(action.astype(jnp.int32), self.num_actions)
        concept_in = jnp.concatenate([z, action_oh, z_next - z], axis=-1)
        return l2_normalize(self.concept_head(concept_in))

    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        next_obs: jnp.ndarray,
    ) -> dict[str, jnp.ndarray]:
        z = self.encode_state(obs)
        z_next = self.encode_state(next_obs)
        action_oh = jax.nn.one_hot(action.astype(jnp.int32), self.num_actions)

        inverse_logits = self.inverse_head(jnp.concatenate([z, z_next], axis=-1))
        pred_next_z = self.forward_head(jnp.concatenate([z, action_oh], axis=-1)) + z

        concept_in = jnp.concatenate([z, action_oh, z_next - z], axis=-1)
        concept = l2_normalize(self.concept_head(concept_in))
        reward_pred = self.reward_head(concept).squeeze(-1)
        done_logit = self.done_head(concept).squeeze(-1)
        delta_pred = self.delta_head(concept)

        return {
            "state": z,
            "next_state": z_next,
            "inverse_logits": inverse_logits,
            "pred_next_state": pred_next_z,
            "concept": concept,
            "reward_pred": reward_pred,
            "done_logit": done_logit,
            "delta_pred": delta_pred,
        }


def info_nce_logits(anchor: jnp.ndarray, positive: jnp.ndarray, temperature: float) -> jnp.ndarray:
    return anchor @ positive.T / temperature


def symmetric_info_nce(anchor: jnp.ndarray, positive: jnp.ndarray, temperature: float) -> jnp.ndarray:
    logits = info_nce_logits(anchor, positive, temperature)
    labels = jnp.arange(logits.shape[0])
    loss_a = optax_softmax_cross_entropy(logits, labels)
    loss_b = optax_softmax_cross_entropy(logits.T, labels)
    return 0.5 * (loss_a + loss_b)


def optax_softmax_cross_entropy(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(log_probs[jnp.arange(labels.shape[0]), labels])
