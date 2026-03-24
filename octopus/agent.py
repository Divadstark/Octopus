

from typing import Any, Mapping, Text, Callable

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

Array = chex.Array
Numeric = chex.Numeric



class Octopus(parts.Agent):

  def __init__(
      self,
      preprocessor: processors.Processor,
      sample_network_input: jnp.ndarray,
      network: parts.Network,
      support: jnp.ndarray,
      optimizer: optax.GradientTransformation,
      transition_accumulator: Any,
      replay: replay_lib.TransitionReplay,
      batch_size: int,
      min_replay_capacity_fraction: float,
      learn_period: int,
      target_network_update_period: int,
      num_actions: int,
      rng_key: parts.PRNGKey,
      tau: float = 0.03,  # explicit entropy temperature
      alpha: float = 0.9,  # scaling of the log-policy
      clip_value_min: float = -1.,# float (<0), minimum value to clip the log-policy
      use_munchausen: bool = True, #toggle for munchausen

  ):

    self._preprocessor = preprocessor
    self._replay = replay
    self._transition_accumulator = transition_accumulator
    self._batch_size = batch_size
    self._min_replay_capacity = min_replay_capacity_fraction * replay.capacity
    self._learn_period = learn_period
    self._target_network_update_period = target_network_update_period
    self.tau = tau
    self.alpha = alpha
    self.clip_value_min = clip_value_min
    self.num_actions = num_actions

    # Initialize network parameters and optimizer.
    self._rng_key, network_rng_key = jax.random.split(rng_key)
    self._online_params = network.init(network_rng_key,
                                       sample_network_input[None, ...])
    self._target_params = self._online_params
    self._opt_state = optimizer.init(self._online_params)

    # Other agent state: last action, frame count, etc.
    self._action = None
    self._frame_t = -1  # Current frame index.
    self._statistics = {'state_value': np.nan}
    self._max_seen_priority = 1.0

    self.current_iteration = 0
    self.use_munchausen = use_munchausen
    self.munchausen_epsilon = float

    def loss_fn(online_params, target_params, transitions, weights, rng_key):
      """Calculates loss given network parameters and transitions."""
      _, *apply_keys = jax.random.split(rng_key, 4)

      logits_q_tm1 = network.apply(
          online_params, apply_keys[0], transitions.s_tm1
      ).q_logits
      q_t = network.apply(
          online_params, apply_keys[1], transitions.s_t
      ).q_values

      q_target_t = network.apply(
          target_params, apply_keys[2], transitions.s_t
      ).q_values
      q_target_tm1 = network.apply(target_params, apply_keys[2],
                                   transitions.s_tm1).q_values
      logits_q_target_t = network.apply(
          target_params, apply_keys[2], transitions.s_t
      ).q_logits

      losses = self.batch_m_categorical_double_q_learning(
          support,
          logits_q_tm1,
          q_target_t,
          q_target_tm1,
          transitions.a_tm1,
          transitions.r_t,
          transitions.discount_t,
          support,
          logits_q_target_t,
          q_t,
      )
      loss = jnp.mean(losses * weights)
      chex.assert_shape((losses, weights), (self._batch_size,))
      return loss, losses

    def update(
        rng_key, opt_state, online_params, target_params, transitions, weights
    ):
      """Computes learning update from batch of replay transitions."""
      rng_key, update_key = jax.random.split(rng_key)
      d_loss_d_params, losses = jax.grad(loss_fn, has_aux=True)(
          online_params, target_params, transitions, weights, update_key
      )
      updates, new_opt_state = optimizer.update(d_loss_d_params, opt_state)
      new_online_params = optax.apply_updates(online_params, updates)
      return rng_key, new_opt_state, new_online_params, losses

    self._update = jax.jit(update)

    def select_action(rng_key, network_params, s_t):
      """Computes greedy (argmax) action wrt Q-values at given state."""
      rng_key, apply_key, policy_key = jax.random.split(rng_key, 3)
      q_t = network.apply(network_params, apply_key, s_t[None, ...]).q_values[0]
      a_t = rlax.greedy().sample(policy_key, q_t)
      v_t = jnp.max(q_t, axis=-1)
      return rng_key, a_t, v_t

    self._select_action = jax.jit(select_action)

  def step(self, timestep: dm_env.TimeStep) -> parts.Action:
    """Selects action given timestep and potentially learns."""
    self._frame_t += 1

    timestep = self._preprocessor(timestep)

    if timestep is None:  # Repeat action.
      action = self._action
    else:
      action = self._action = self._act(timestep)

      for transition in self._transition_accumulator.step(timestep, action):
        self._replay.add(transition, priority=self._max_seen_priority)

    if self._replay.size < self._min_replay_capacity:
      return action

    if self._frame_t % self._learn_period == 0:
      self._learn()

    if self._frame_t % self._target_network_update_period == 0:
      self._target_params = self._online_params

    return action

  def reset(self) -> None:
    """Resets the agent's episodic state such as frame stack and action repeat.

    This method should be called at the beginning of every episode.
    """
    self._transition_accumulator.reset()
    processors.reset(self._preprocessor)
    self._action = None

  def _act(self, timestep) -> parts.Action:
    """Selects action given timestep, according to greedy policy."""
    s_t = timestep.observation
    self._rng_key, a_t, v_t = self._select_action(
        self._rng_key, self._online_params, s_t
    )
    a_t, v_t = jax.device_get((a_t, v_t))
    self._statistics['state_value'] = v_t
    return parts.Action(a_t)

  def _learn(self) -> None:
    """Samples a batch of transitions from replay and learns from it."""
    logging.log_first_n(logging.INFO, 'Begin learning', 1)
    transitions, indices, weights = self._replay.sample(self._batch_size)
    self._rng_key, self._opt_state, self._online_params, losses = self._update(
        self._rng_key,
        self._opt_state,
        self._online_params,
        self._target_params,
        transitions,
        weights,
    )
    chex.assert_equal_shape((losses, weights))
    priorities = jnp.clip(jnp.abs(losses), 0.0, 100.0)
    priorities = jax.device_get(priorities)
    max_priority = priorities.max()
    self._max_seen_priority = np.max([self._max_seen_priority, max_priority])
    self._replay.update_priorities(indices, priorities)

  def update_munchausen_epsilon(self,current_iteration):
      if current_iteration >= 30:
          step_size = 0.7 / 170
          steps_into_increase_phase = current_iteration - 30
          self.munchausen_epsilon = steps_into_increase_phase * step_size
          if self.munchausen_epsilon > 0.7:
              self.munchausen_epsilon = 0.7
      else:
          self.munchausen_epsilon = 0



  @property
  def online_params(self) -> parts.NetworkParams:
    """Returns current parameters of Q-network."""
    return self._online_params

  @property
  def statistics(self) -> Mapping[Text, float]:
    """Returns current agent statistics as a dictionary."""
    # Check for DeviceArrays in values as this can be very slow.
    assert all(
        not isinstance(x, jnp.DeviceArray) for x in self._statistics.values()
    )
    return self._statistics

  @property
  def importance_sampling_exponent(self) -> float:
    """Returns current importance sampling exponent of prioritized replay."""
    return self._replay.importance_sampling_exponent

  @property
  def max_seen_priority(self) -> float:
    """Returns maximum seen replay priority up until this time."""
    return self._max_seen_priority

  def get_state(self) -> Mapping[Text, Any]:
    """Retrieves agent state as a dictionary (e.g. for serialization)."""
    state = {
        'rng_key': self._rng_key,
        'frame_t': self._frame_t,
        'opt_state': self._opt_state,
        'online_params': self._online_params,
        'target_params': self._target_params,
        'replay': self._replay.get_state(),
        'max_seen_priority': self._max_seen_priority,
    }
    return state

  def set_state(self, state: Mapping[Text, Any]) -> None:
    """Sets agent state from a (potentially de-serialized) dictionary."""
    self._rng_key = state['rng_key']
    self._frame_t = state['frame_t']
    self._opt_state = jax.device_put(state['opt_state'])
    self._online_params = jax.device_put(state['online_params'])
    self._target_params = jax.device_put(state['target_params'])
    self._replay.set_state(state['replay'])
    self._max_seen_priority = state['max_seen_priority']

  # Implements the Octopus distributional Bellman update:
  #   Ẑ_oct = r + ατ log π(a | s) + γ [Z(s', a') − τ log π(a' | s')]
  # See Eq. (3) and Section 4 in the Octopus paper for theoretical details.
  # This target is projected using C51-style L2 projection onto fixed atom supports.
  def m_categorical_double_q_learning(
          self,
          q_atoms_tm1: Array,
          q_logits_tm1: Array,
          q_target_tm1: Array,
          q_target_t: Array,
          a_tm1: Numeric,
          r_t: Numeric,
          discount_t: Numeric,
          q_atoms_t: Array,
          q_logits_t: Array,
          q_t_selector: Array,
          stop_target_gradients: bool = True,
  ) -> Numeric:

      chex.assert_rank([
          q_atoms_tm1, q_logits_tm1, q_target_tm1, q_target_t, a_tm1, r_t, discount_t, q_atoms_t, q_logits_t,
          q_t_selector
      ], [1, 2, 1, 1, 0, 0, 0, 1, 2, 1])
      chex.assert_type([
          q_atoms_tm1, q_logits_tm1, q_target_tm1, q_target_t, a_tm1, r_t, discount_t, q_atoms_t, q_logits_t,
          q_t_selector
      ], [float, float, float, float, int, float, float, float, float, float])
      munchausen_term = self.calculate_munchausen_term(q_target_tm1, a_tm1)
      next_log_policy = self.stable_scaled_log_softmax(q_target_t, self.tau, axis=-1)
      next_policy = self.stable_softmax(q_target_t, self.tau, axis=-1)
      next_tau_log_pi = jnp.sum((next_log_policy) * next_policy, axis=-1)

      #Scale and shift time-t distribution atoms by discount and reward.
      if self.use_munchausen == True:
          target_z = r_t + self.munchausen_epsilon * (munchausen_term - discount_t * next_tau_log_pi) + discount_t * q_atoms_t
      else:
          target_z = r_t + discount_t * q_atoms_t
      #target_orginal_z = r_t + discount_t * q_atoms_t
      p_target_z = jax.nn.softmax(q_logits_t[q_t_selector.argmax()])
      # Project using the Cramer distance and maybe stop gradient flow to targets.
      target = rlax.categorical_l2_project(target_z, p_target_z, q_atoms_tm1)
      target = jax.lax.select(stop_target_gradients, jax.lax.stop_gradient(target),
                              target)

      # Compute loss (i.e. temporal difference error).
      logit_qa_tm1 = q_logits_tm1[a_tm1]
      return rlax.categorical_cross_entropy(
          labels=target, logits=logit_qa_tm1)

  # Implements the Munchausen reward term: α * τ * log π(a_t | s_t)
  # Refer to Eq. (3) and Section 4 in the Octopus paper.
  def calculate_munchausen_term(self, q_target_tm1, a_tm1):
      # Compute the softmax probabilities over the q_logits to get Q-value distribution
      action_one_hot = jax.nn.one_hot(a_tm1, self.num_actions)
      # tau * ln pi_{k+1} (s)
      log_policy = self.stable_scaled_log_softmax(q_target_tm1, self.tau, axis=-1)
       # tau * ln pi_{k+1} (a|s)
      tau_log_pi_a = jnp.sum(log_policy * action_one_hot, axis=-1)
      tau_log_pi_a = jnp.clip(tau_log_pi_a, a_min=self.clip_value_min, a_max=1)

      munchausen_term = self.alpha * tau_log_pi_a

      return munchausen_term

  def stable_scaled_log_softmax(self, x, tau, axis=-1):

      max_x = jnp.max(x, axis=axis, keepdims=True)
      y = x - max_x
      tau_lse = max_x + tau * jnp.log(
          jnp.sum(jnp.exp(y / tau), axis=axis, keepdims=True))
      return x - tau_lse

  def stable_softmax(self, x, tau, axis=-1):

      max_x = jnp.max(x, axis=axis, keepdims=True)
      y = x - max_x
      return jax.nn.softmax(y / tau, axis=axis)

  def batch_m_categorical_double_q_learning(self,
      q_atoms_tm1,
      q_logits_tm1,
      q_target_tm1,
      q_target_t,
      a_tm1,
      r_t,
      discount_t,
      q_atoms_t,
      q_logits_t,
      q_t_selector):

      vmapped_func = jax.vmap(self.m_categorical_double_q_learning, in_axes=(None, 0, 0, 0, 0, 0, 0, None, 0, 0))
      return vmapped_func(
          q_atoms_tm1,
          q_logits_tm1,
          q_target_tm1,
          q_target_t,
          a_tm1,
          r_t,
          discount_t,
          q_atoms_t,
          q_logits_t,
          q_t_selector,)
