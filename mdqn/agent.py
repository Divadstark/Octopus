
from typing import Any, Callable, Mapping, Text

from absl import logging
import chex
import dm_env
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax

from dqn_zoo import parts
from dqn_zoo import processors
from dqn_zoo import replay as replay_lib

def mq_learning(q_tm1,
a_tm1, r_t, discount_t, munchausen_term, next_qt_softmax):
  target_tm1 = r_t + munchausen_term + discount_t * next_qt_softmax
  td_errors = jax.lax.stop_gradient(target_tm1) - q_tm1[a_tm1]
  return td_errors

_batch_mq_learning = jax.vmap(mq_learning)


class MDqn(parts.Agent):
  

  def __init__(
      self,
      preprocessor: processors.Processor,
      sample_network_input: jnp.ndarray,
      network: parts.Network,
      optimizer: optax.GradientTransformation,
      transition_accumulator: Any,
      replay: replay_lib.TransitionReplay,
      batch_size: int,
      exploration_epsilon: Callable[[int], float],
      min_replay_capacity_fraction: float,
      learn_period: int,
      target_network_update_period: int,
      grad_error_bound: float,
      num_actions: int,
      rng_key: parts.PRNGKey,
      tau: float = 0.03,  # explicit entropy temperature
      alpha: float = 0.9,  # scaling of the log-policy
      clip_value_min: float = -1.  # float (<0), minimum value to clip the log-policy
  ):
    self._preprocessor = preprocessor
    self._replay = replay
    self._transition_accumulator = transition_accumulator
    self._batch_size = batch_size
    self._exploration_epsilon = exploration_epsilon
    self._min_replay_capacity = min_replay_capacity_fraction * replay.capacity
    self._learn_period = learn_period
    self._target_network_update_period = target_network_update_period
    self.tau = tau
    self.alpha = alpha
    self.clip_value_min = clip_value_min

    self._rng_key, network_rng_key = jax.random.split(rng_key)
    self._online_params = network.init(network_rng_key,
                                       sample_network_input[None, ...])
    self._target_params = self._online_params
    self._opt_state = optimizer.init(self._online_params)

    self._action = None
    self._frame_t = -1  # Current frame index.
    self._statistics = {'state_value': np.nan}


    def loss_fn(online_params, target_params, transitions, rng_key):
      
      _, online_key, target_key = jax.random.split(rng_key, 3)
      q_tm1 = network.apply(online_params, online_key,
                            transitions.s_tm1).q_values
      q_target_t = network.apply(target_params, target_key,
                                 transitions.s_t).q_values
      q_target_tm1 = network.apply(target_params, target_key,
                                 transitions.s_tm1).q_values

      action_one_hot = jax.nn.one_hot(transitions.a_tm1, num_actions)
      next_log_policy = self.stable_scaled_log_softmax(q_target_t, self.tau, axis=-1)
      log_policy = self.stable_scaled_log_softmax(q_target_tm1, self.tau, axis=-1)
      next_policy = self.stable_softmax(q_target_t, self.tau, axis=-1)

      next_qt_softmax = jnp.sum((q_target_t - next_log_policy) * next_policy, axis=-1)
      tau_log_pi_a = jnp.sum(log_policy * action_one_hot, axis=-1)
      tau_log_pi_a = jnp.clip(tau_log_pi_a, a_min=self.clip_value_min, a_max=1)

      munchausen_term = self.alpha * tau_log_pi_a

      td_errors = _batch_mq_learning(
        q_tm1,
        transitions.a_tm1,
        transitions.r_t,
        transitions.discount_t,
        munchausen_term,
        next_qt_softmax
      )

      td_errors = rlax.clip_gradient(td_errors, -grad_error_bound,
                                     grad_error_bound)
      losses = rlax.l2_loss(td_errors)
      chex.assert_shape(losses, (self._batch_size,))
      loss = jnp.mean(losses)
      return loss

    def update(rng_key, opt_state, online_params, target_params, transitions):
      
      rng_key, update_key = jax.random.split(rng_key)
      d_loss_d_params = jax.grad(loss_fn)(online_params, target_params,
                                          transitions, update_key)
      updates, new_opt_state = optimizer.update(d_loss_d_params, opt_state)
      new_online_params = optax.apply_updates(online_params, updates)
      return rng_key, new_opt_state, new_online_params

    self._update = jax.jit(update)

    def select_action(rng_key, network_params, s_t, exploration_epsilon):
      
      rng_key, apply_key, policy_key, epsilon_key, action_key = jax.random.split(rng_key, 5)
      q_t = network.apply(network_params, apply_key, s_t[None, ...]).q_values[0]
      policy_logit = self.stable_scaled_log_softmax(q_t, self.tau, axis=-1) / self.tau
      a_t = jax.random.categorical(policy_key, policy_logit, axis=-1)
      v_t = jnp.max(q_t, axis=-1)
      return rng_key, a_t, v_t

    self._select_action = jax.jit(select_action)  # jax.jit(select_action)

  def step(self, timestep: dm_env.TimeStep) -> parts.Action:
    
    self._frame_t += 1

    timestep = self._preprocessor(timestep)

    if timestep is None:  # Repeat action.
      action = self._action
    else:
      action = self._action = self._act(timestep)

      for transition in self._transition_accumulator.step(timestep, action):
        self._replay.add(transition)

    if self._replay.size < self._min_replay_capacity:
      return action

    if self._frame_t % self._learn_period == 0:
      self._learn()

    if self._frame_t % self._target_network_update_period == 0:
      self._target_params = self._online_params

    return action

  def reset(self) -> None:
    
    self._transition_accumulator.reset()
    processors.reset(self._preprocessor)
    self._action = None

  def _act(self, timestep) -> parts.Action:
    
    s_t = timestep.observation
    self._rng_key, a_t, v_t = self._select_action(self._rng_key,
                                                  self._online_params, s_t,
                                                  self.exploration_epsilon)
    a_t, v_t = jax.device_get((a_t, v_t))
    self._statistics['state_value'] = v_t
    return parts.Action(a_t)

  def _learn(self) -> None:
    
    logging.log_first_n(logging.INFO, 'Begin learning', 1)
    transitions = self._replay.sample(self._batch_size)
    self._rng_key, self._opt_state, self._online_params = self._update(
        self._rng_key,
        self._opt_state,
        self._online_params,
        self._target_params,
        transitions,
    )

  @property
  def online_params(self) -> parts.NetworkParams:
    
    return self._online_params

  @property
  def statistics(self) -> Mapping[Text, float]:
    
    assert all(
        not isinstance(x, jnp.DeviceArray) for x in self._statistics.values())
    return self._statistics

  @property
  def exploration_epsilon(self) -> float:
    
    return self._exploration_epsilon(self._frame_t)

  def get_state(self) -> Mapping[Text, Any]:
    
    state = {
        'rng_key': self._rng_key,
        'frame_t': self._frame_t,
        'opt_state': self._opt_state,
        'online_params': self._online_params,
        'target_params': self._target_params,
        'replay': self._replay.get_state(),
    }
    return state

  def set_state(self, state: Mapping[Text, Any]) -> None:
    
    self._rng_key = state['rng_key']
    self._frame_t = state['frame_t']
    self._opt_state = jax.device_put(state['opt_state'])
    self._online_params = jax.device_put(state['online_params'])
    self._target_params = jax.device_put(state['target_params'])
    self._replay.set_state(state['replay'])

  def stable_scaled_log_softmax(self, x, tau, axis=-1):
    
    max_x = jnp.max(x, axis=axis, keepdims=True)
    y = x - max_x
    tau_lse = max_x + tau * jnp.log(
      jnp.sum(jnp.exp(y / tau), axis=axis, keepdims=True))
    return x - tau_lse

  def stable_softmax(self, x, tau, axis=-1):
    
    max_x = jnp.max(x, axis=axis, keepdims=True)
    y = x - max_x
    return jax.nn.softmax(y/tau, axis=axis)
