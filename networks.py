
import typing
from typing import Any, Callable, Tuple, Union

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

Network = hk.Transformed
Params = hk.Params
NetworkFn = Callable[..., Any]


class QNetworkOutputs(typing.NamedTuple):
  q_values: jnp.ndarray


class IqnInputs(typing.NamedTuple):
  state: jnp.ndarray
  taus: jnp.ndarray


class IqnOutputs(typing.NamedTuple):
  q_values: jnp.ndarray
  q_dist: jnp.ndarray


class QRNetworkOutputs(typing.NamedTuple):
  q_values: jnp.ndarray
  q_dist: jnp.ndarray


class C51NetworkOutputs(typing.NamedTuple):
  q_values: jnp.ndarray
  q_logits: jnp.ndarray


def _dqn_default_initializer(
    num_input_units: int,
) -> hk.initializers.Initializer:
  
  max_val = np.sqrt(1 / num_input_units)
  return hk.initializers.RandomUniform(-max_val, max_val)


def conv(
    num_features: int,
    kernel_shape: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]],
) -> NetworkFn:
  

  def net_fn(inputs):
    
    num_input_units = inputs.shape[-1] * kernel_shape[0] * kernel_shape[1] #64
    initializer = _dqn_default_initializer(num_input_units)
    layer = hk.Conv2D(
        num_features,
        kernel_shape=kernel_shape,
        stride=stride,
        w_init=initializer,
        b_init=initializer,
        padding='VALID',
    )
    return layer(inputs)

  return net_fn


def linear(num_outputs: int, with_bias=True) -> NetworkFn:
  

  def net_fn(inputs):
    
    initializer = _dqn_default_initializer(inputs.shape[-1])
    layer = hk.Linear(
        num_outputs, with_bias=with_bias, w_init=initializer, b_init=initializer
    )
    return layer(inputs)

  return net_fn


def linear_with_shared_bias(num_outputs: int) -> NetworkFn:
  

  def layer_fn(inputs):
    
    initializer = _dqn_default_initializer(inputs.shape[-1])
    bias_free_linear = hk.Linear(
        num_outputs, with_bias=False, w_init=initializer
    )
    linear_output = bias_free_linear(inputs)
    bias = hk.get_parameter('b', [1], inputs.dtype, init=initializer)
    bias = jnp.broadcast_to(bias, linear_output.shape)
    return linear_output + bias

  return layer_fn


def noisy_linear(
    num_outputs: int, weight_init_stddev: float, with_bias: bool = True
) -> NetworkFn:
  

  def make_noise_sqrt(rng, shape):
    noise = jax.random.truncated_normal(rng, lower=-2.0, upper=2.0, shape=shape)
    return jax.lax.stop_gradient(jnp.sign(noise) * jnp.sqrt(jnp.abs(noise)))

  def net_fn(inputs):
    
    num_inputs = inputs.shape[-1]
    mu_initializer = _dqn_default_initializer(num_inputs)
    mu_layer = hk.Linear(
        num_outputs,
        name='mu',
        with_bias=with_bias,
        w_init=mu_initializer,
        b_init=mu_initializer,
    )
    sigma_initializer = hk.initializers.Constant(  #
        weight_init_stddev / jnp.sqrt(num_inputs)
    )
    sigma_layer = hk.Linear(
        num_outputs,
        name='sigma',
        with_bias=True,
        w_init=sigma_initializer,
        b_init=sigma_initializer,
    )

    input_noise_sqrt = make_noise_sqrt(hk.next_rng_key(), [1, num_inputs])
    output_noise_sqrt = make_noise_sqrt(hk.next_rng_key(), [1, num_outputs])

    mu = mu_layer(inputs)
    noisy_inputs = input_noise_sqrt * inputs
    sigma = sigma_layer(noisy_inputs) * output_noise_sqrt
    return mu + sigma

  return net_fn


def dqn_torso() -> NetworkFn:
  

  def net_fn(inputs):
    
    network = hk.Sequential([
        lambda x: x.astype(jnp.float32) / 255.0,
        conv(32, kernel_shape=(8, 8), stride=(4, 4)),
        jax.nn.relu,
        conv(64, kernel_shape=(4, 4), stride=(2, 2)),
        jax.nn.relu,
        conv(64, kernel_shape=(3, 3), stride=(1, 1)),
        jax.nn.relu,
        hk.Flatten(),
    ])
    return network(inputs)

  return net_fn


def dqn_value_head(num_actions: int, shared_bias: bool = False) -> NetworkFn:
  

  last_layer = linear_with_shared_bias if shared_bias else linear

  def net_fn(inputs):
    
    network = hk.Sequential([
        linear(512),
        jax.nn.relu,
        last_layer(num_actions),
    ])
    return network(inputs)

  return net_fn


def rainbow_atari_network(
    num_actions: int,
    support: jnp.ndarray,
    noisy_weight_init: float,
) -> NetworkFn:
  

  chex.assert_rank(support, 1)
  num_atoms = len(support)
  support = support[None, None, :]

  def net_fn(inputs):
    
    inputs = dqn_torso()(inputs)

    advantage = noisy_linear(512, noisy_weight_init, with_bias=True)(inputs)
    advantage = jax.nn.relu(advantage)
    advantage = noisy_linear(
        num_actions * num_atoms, noisy_weight_init, with_bias=False
    )(advantage)
    advantage = jnp.reshape(advantage, (-1, num_actions, num_atoms))

    value = noisy_linear(512, noisy_weight_init, with_bias=True)(inputs)
    value = jax.nn.relu(value)
    value = noisy_linear(num_atoms, noisy_weight_init, with_bias=False)(value)
    value = jnp.reshape(value, (-1, 1, num_atoms))

    q_logits = value + advantage - jnp.mean(advantage, axis=-2, keepdims=True)
    assert q_logits.shape[1:] == (num_actions, num_atoms)
    q_dist = jax.nn.softmax(q_logits)
    q_values = jnp.sum(q_dist * support, axis=2)
    q_values = jax.lax.stop_gradient(q_values)
    return C51NetworkOutputs(q_logits=q_logits, q_values=q_values)

  return net_fn


def iqn_atari_network(num_actions: int, latent_dim: int) -> NetworkFn:
  

  def net_fn(iqn_inputs):
    
    state = iqn_inputs.state  # batch x state_shape
    taus = iqn_inputs.taus  # batch x samples
    state_embedding = dqn_torso()(state)
    state_dim = state_embedding.shape[-1]
    pi_multiples = jnp.arange(1, latent_dim + 1, dtype=jnp.float32) * jnp.pi
    tau_embedding = jnp.cos(pi_multiples[None, None, :] * taus[:, :, None])
    embedding_layer = linear(state_dim)
    tau_embedding = hk.BatchApply(embedding_layer)(tau_embedding)
    tau_embedding = jax.nn.relu(tau_embedding)
    head_input = tau_embedding * state_embedding[:, None, :]
    value_head = dqn_value_head(num_actions)
    q_dist = hk.BatchApply(value_head)(head_input)
    q_values = jnp.mean(q_dist, axis=1)
    q_values = jax.lax.stop_gradient(q_values)
    return IqnOutputs(q_dist=q_dist, q_values=q_values)

  return net_fn


def qr_atari_network(num_actions: int, quantiles: jnp.ndarray) -> NetworkFn:
  

  chex.assert_rank(quantiles, 1)
  num_quantiles = len(quantiles)

  def net_fn(inputs):
    
    network = hk.Sequential([
        dqn_torso(),
        dqn_value_head(num_quantiles * num_actions),
    ])
    network_output = network(inputs)
    q_dist = jnp.reshape(network_output, (-1, num_quantiles, num_actions))
    q_values = jnp.mean(q_dist, axis=1)
    q_values = jax.lax.stop_gradient(q_values)
    return QRNetworkOutputs(q_dist=q_dist, q_values=q_values)

  return net_fn


def c51_atari_network(num_actions: int, support: jnp.ndarray) -> NetworkFn:
  

  chex.assert_rank(support, 1)
  num_atoms = len(support)

  def net_fn(inputs):
    
    network = hk.Sequential([
        dqn_torso(),
        dqn_value_head(num_actions * num_atoms),
    ])
    network_output = network(inputs)
    q_logits = jnp.reshape(network_output, (-1, num_actions, num_atoms))
    q_dist = jax.nn.softmax(q_logits)
    q_values = jnp.sum(q_dist * support[None, None, :], axis=2)
    q_values = jax.lax.stop_gradient(q_values)
    return C51NetworkOutputs(q_logits=q_logits, q_values=q_values)

  return net_fn


def double_dqn_atari_network(num_actions: int) -> NetworkFn:
  

  def net_fn(inputs):
    
    network = hk.Sequential([
        dqn_torso(),
        dqn_value_head(num_actions, shared_bias=True),
    ])
    return QNetworkOutputs(q_values=network(inputs))

  return net_fn


def dqn_atari_network(num_actions: int) -> NetworkFn:
  

  def net_fn(inputs):
    
    network = hk.Sequential([
        dqn_torso(),
        dqn_value_head(num_actions),
    ])
    return QNetworkOutputs(q_values=network(inputs))

  return net_fn


def rainbow_atari_network_nonoise(
    num_actions: int,
    support: jnp.ndarray,
) -> NetworkFn:
  

  chex.assert_rank(support, 1)
  num_atoms = len(support)
  support = support[None, None, :]

  def net_fn(inputs):
    
    inputs = dqn_torso()(inputs)

    advantage = linear(512, with_bias=True)(inputs)
    advantage = jax.nn.relu(advantage)
    advantage = linear(
      num_actions * num_atoms, with_bias=False
    )(advantage)
    advantage = jnp.reshape(advantage, (-1, num_actions, num_atoms))
    value = linear(512, with_bias=True)(inputs)
    value = jax.nn.relu(value)
    value = linear(num_atoms, with_bias=False)(value)
    value = jnp.reshape(value, (-1, 1, num_atoms))

    q_logits = value + advantage - jnp.mean(advantage, axis=-2, keepdims=True)
    assert q_logits.shape[1:] == (num_actions, num_atoms)
    q_dist = jax.nn.softmax(q_logits)
    q_values = jnp.sum(q_dist * support, axis=2)
    q_values = jax.lax.stop_gradient(q_values)
    return C51NetworkOutputs(q_logits=q_logits, q_values=q_values)

  return net_fn


def miqn_atari_network(num_actions: int, latent_dim: int, weight_init_stddev: float = 0.5) -> NetworkFn:
  

  def net_fn(iqn_inputs):
    state = iqn_inputs.state  # Shape: [batch_size, H, W, C]
    taus = iqn_inputs.taus  # Shape: [batch_size, num_taus]
    batch_size = state.shape[0]
    num_taus = taus.shape[1]

    state_embedding = dqn_torso()(state)  # Shape: [batch_size, embedding_dim]
    embedding_dim = state_embedding.shape[-1]

    pi_multiples = jnp.arange(1, latent_dim + 1, dtype=jnp.float32) * jnp.pi
    tau_embedding = jnp.cos(pi_multiples[None, None, :] * taus[:, :, None])  # Shape: [batch_size, num_taus, latent_dim]

    embedding_layer = linear(embedding_dim, with_bias=False)
    tau_embedding = hk.BatchApply(embedding_layer)(tau_embedding)  # Shape: [batch_size, num_taus, embedding_dim]
    tau_embedding = jax.nn.relu(tau_embedding)

    state_embedding = state_embedding[:, None, :]  # Shape: [batch_size, 1, embedding_dim]
    state_embedding = jnp.broadcast_to(state_embedding,
                                       (batch_size, num_taus, embedding_dim))  # Now matches tau_embedding's shape

    combined_embedding = state_embedding * tau_embedding  # Shape: [batch_size, num_taus, embedding_dim]

    combined_embedding = combined_embedding.reshape(-1, embedding_dim)  # Shape: [batch_size * num_taus, embedding_dim]

    hidden = noisy_linear(512, weight_init_stddev, with_bias=True)(combined_embedding)
    hidden = jax.nn.relu(hidden)

    advantage = noisy_linear(512, weight_init_stddev, with_bias=True)(hidden)
    advantage = jax.nn.relu(advantage)
    advantage = noisy_linear(num_actions, weight_init_stddev, with_bias=False)(advantage)

    value = noisy_linear(512, weight_init_stddev, with_bias=True)(hidden)
    value = jax.nn.relu(value)
    value = noisy_linear(1, weight_init_stddev, with_bias=False)(value)

    advantage = advantage.reshape(batch_size, num_taus, num_actions)
    value = value.reshape(batch_size, num_taus, 1)

    q_values = value + (
              advantage - jnp.mean(advantage, axis=2, keepdims=True))  # Shape: [batch_size, num_taus, num_actions]

    q_values_mean = jnp.mean(q_values, axis=1)  # Shape: [batch_size, num_actions]
    q_values_mean = jax.lax.stop_gradient(q_values_mean)

    return IqnOutputs(q_values=q_values_mean, q_dist=q_values)

  return net_fn