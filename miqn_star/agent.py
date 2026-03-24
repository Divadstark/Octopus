

from typing import Any, Callable, Mapping, Text, Tuple

from absl import logging
import chex
import distrax
import dm_env
import jax
import jax.numpy as jnp
import numpy as np
import optax

from dqn_zoo import networks
from dqn_zoo import parts
from dqn_zoo import replay as replay_lib
from dqn_zoo import processors

Array = chex.Array
Numeric = chex.Numeric

IqnInputs = networks.IqnInputs


def _sample_tau(
        rng_key: parts.PRNGKey,
        shape: Tuple[int, ...],
) -> jnp.ndarray:
    
    return jax.random.uniform(rng_key, shape=shape)


class miqn_star(parts.Agent):
    

    def __init__(
            self,
            preprocessor: processors.Processor,
            sample_network_input: IqnInputs,
            network: parts.Network,
            optimizer: optax.GradientTransformation,
            transition_accumulator: Any,  # Should be a MultiStepTransitionAccumulator
            replay: replay_lib.PrioritizedTransitionReplay,  # Prioritized Replay Buffer
            batch_size: int,
            exploration_epsilon: Callable[[int], float],
            min_replay_capacity_fraction: float,
            learn_period: int,
            target_network_update_period: int,
            huber_param: float,
            tau_samples_policy: int,
            tau_samples_s_tm1: int,
            tau_samples_s_t: int,
            rng_key: parts.PRNGKey,
            tau: float = 0.03,
            alpha: float = 0.9,
            clip_value_min: float = -1,
            multi_step: int = 3,  # Number of steps for multi-step learning
            gamma: float = 0.99,  # Discount factor
    ):
        self._preprocessor = preprocessor
        self._replay = replay
        self._transition_accumulator = transition_accumulator
        self._batch_size = batch_size
        self._exploration_epsilon = exploration_epsilon
        self._min_replay_capacity = min_replay_capacity_fraction * replay.capacity
        self._learn_period = learn_period
        self._target_network_update_period = target_network_update_period

        self._huber_param = huber_param
        self._tau_samples_policy = tau_samples_policy
        self._tau_samples_s_tm1 = tau_samples_s_tm1
        self._tau_samples_s_t = tau_samples_s_t

        self._tau = tau
        self._alpha = alpha
        self._clip_value_min = clip_value_min
        self._multi_step = multi_step
        self._gamma = gamma

        self._rng_key, network_rng_key = jax.random.split(rng_key)
        self._online_params = network.init(
            network_rng_key,
            jax.tree_map(lambda x: x[None, ...], sample_network_input),
        )
        self._target_params = self._online_params
        self._opt_state = optimizer.init(self._online_params)

        self._action = None
        self._frame_t = -1  # Current frame index.
        self._statistics = {'state_value': np.nan, 'loss': np.nan}

        self._max_seen_priority = 1.0
        self.munchausen_epsilon = float

        def stable_scaled_log_softmax(x, tau, axis=-1):
            
            max_x = jnp.max(x, axis=axis, keepdims=True)
            y = x - max_x
            tau_lse = max_x + tau * jnp.log(
                jnp.sum(jnp.exp(y / tau), axis=axis, keepdims=True)
            )
            return x - tau_lse

        def stable_softmax(x, tau, axis=-1):
            
            max_x = jnp.max(x, axis=axis, keepdims=True)
            y = x - max_x
            return jax.nn.softmax(y / tau, axis=axis)

        def quantile_regression_loss(
                dist_src: Array,
                tau_src: Array,
                dist_target: Array,
                huber_param: float = 0.,
                stop_target_gradients: bool = True,
        ) -> Numeric:
            
            delta = dist_target[None, :] - dist_src[:, None]  # Shape: [num_tau_src, num_tau_target]
            delta_neg = (delta < 0.).astype(jnp.float32)
            if stop_target_gradients:
                delta_neg = jax.lax.stop_gradient(delta_neg)
            weight = jnp.abs(tau_src[:, None] - delta_neg)  # Shape: [num_tau_src, num_tau_target]

            if huber_param > 0.:
                loss = huber_loss(delta, huber_param)
            else:
                loss = jnp.abs(delta)
            loss *= weight

            return jnp.sum(jnp.mean(loss, axis=-1))  # Scalar

        def huber_loss(delta, kappa):
            
            abs_delta = jnp.abs(delta)
            quadratic = jnp.minimum(abs_delta, kappa)
            linear = abs_delta - quadratic
            loss = 0.5 * quadratic ** 2 + kappa * linear
            return loss

        def miqn_quantile_q_learning(
                dist_q_tm1: Array,  # [num_tau_samples_s_tm1, num_actions]
                tau_q_tm1: Array,  # [num_tau_samples_s_tm1]
                a_tm1: int,  # Scalar
                r_t: float,  # Scalar
                discount_t: float,  # Scalar
                dist_q_t: Array,  # [num_tau_samples_s_t, num_actions]
                q_tm1: Array,  # [num_actions]
                q_t_selector: Array,  # [num_actions]
                q_t: Array,  # [num_actions]
                num_actions: int,
                tau: float,
                alpha: float,
                clip_value_min: float,
                huber_param: float = 0.0,
                stop_target_gradients: bool = True,
        ) -> float:
            

            dist_qa_tm1 = dist_q_tm1[:, a_tm1]  # [num_tau_samples_s_tm1]

            log_policy_tm1 = stable_scaled_log_softmax(q_tm1, tau, axis=-1)  # [num_actions]
            action_one_hot = jax.nn.one_hot(a_tm1, num_actions)  # [num_actions]
            tau_log_pi_a = jnp.sum(log_policy_tm1 * action_one_hot, axis=-1)  # Scalar
            tau_log_pi_a = jnp.clip(tau_log_pi_a, a_min=clip_value_min, a_max=0.0)
            munchausen_term = alpha * tau_log_pi_a  # Scalar

            log_policy_t = stable_scaled_log_softmax(q_t_selector, tau, axis=-1)  # [num_actions]
            policy_t = stable_softmax(q_t_selector, tau, axis=-1)  # [num_actions]
            next_tau_log_pi = jnp.sum(policy_t * log_policy_t)
            q_atoms_t = jnp.sum(policy_t[None, :] * dist_q_t, axis=-1)  # [num_tau_samples_s_t]
            dist_target = r_t + self.munchausen_epsilon * (
                        munchausen_term - discount_t * next_tau_log_pi) + discount_t * q_atoms_t

            if stop_target_gradients:
                dist_target = jax.lax.stop_gradient(dist_target)

            loss = quantile_regression_loss(
                dist_src=dist_qa_tm1,
                tau_src=tau_q_tm1,
                dist_target=dist_target,
                huber_param=huber_param
            )
            return loss

        def loss_fn(online_params, target_params, transitions, weights, rng_key):
            
            batch_size = self._batch_size

            rng_key, *sample_keys = jax.random.split(rng_key, 4)
            tau_tm1 = _sample_tau(sample_keys[0], (batch_size, self._tau_samples_s_tm1))
            tau_t_selector = _sample_tau(sample_keys[1], (batch_size, self._tau_samples_policy))
            tau_t = _sample_tau(sample_keys[2], (batch_size, self._tau_samples_s_t))

            dist_q_tm1 = network.apply(
                online_params, sample_keys[0], IqnInputs(transitions.s_tm1, tau_tm1)
            ).q_dist
            dist_q_t_selector = network.apply(
                online_params, sample_keys[1], IqnInputs(transitions.s_t, tau_t_selector)
            ).q_dist

            dist_q_t = network.apply(
                target_params, sample_keys[2], IqnInputs(transitions.s_t, tau_t)
            ).q_dist

            q_tm1 = jnp.mean(dist_q_tm1, axis=1)
            q_t_selector = jnp.mean(dist_q_t_selector, axis=1)
            q_t = jnp.mean(dist_q_t, axis=1)

            num_actions = q_tm1.shape[1]

            a_tm1 = transitions.a_tm1
            r_t = transitions.r_t
            discount_t = transitions.discount_t

            per_sample_losses = jax.vmap(
                miqn_quantile_q_learning,
                in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None, None, None, None)
            )(
                dist_q_tm1,
                tau_tm1,
                a_tm1,
                r_t,
                discount_t,
                dist_q_t,
                q_tm1,
                q_t_selector,
                q_t,
                num_actions,
                self._tau,
                self._alpha,
                self._clip_value_min,
                self._huber_param,
                True,
            )

            weighted_losses = per_sample_losses * weights
            loss = jnp.mean(weighted_losses)
            return loss, per_sample_losses

        def update(
                rng_key, opt_state, online_params, target_params, transitions, weights
        ):
            
            rng_key, update_key = jax.random.split(rng_key)
            d_loss_d_params, losses = jax.grad(loss_fn, has_aux=True)(
                online_params, target_params, transitions, weights, update_key
            )
            updates, new_opt_state = optimizer.update(d_loss_d_params, opt_state)
            new_online_params = optax.apply_updates(online_params, updates)
            return rng_key, new_opt_state, new_online_params, losses

        self._update = jax.jit(update)

        def select_action(rng_key, network_params, s_t, exploration_epsilon):
            
            rng_key, sample_key, apply_key, policy_key = jax.random.split(rng_key, 4)
            tau_t = _sample_tau(sample_key, (1, self._tau_samples_policy))
            q_t = network.apply(
                network_params, apply_key, IqnInputs(s_t[None, ...], tau_t)
            ).q_values[0]  # Shape: [num_actions]
            a_t = distrax.EpsilonGreedy(q_t, exploration_epsilon).sample(
                seed=policy_key
            )
            v_t = jnp.max(q_t, axis=-1)
            return rng_key, a_t, v_t

        self._select_action = jax.jit(select_action)

    def step(self, timestep: dm_env.TimeStep) -> parts.Action:
        
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
        
        self._transition_accumulator.reset()
        processors.reset(self._preprocessor)
        self._action = None

    def _act(self, timestep) -> parts.Action:
        
        s_t = timestep.observation
        self._rng_key, a_t, v_t = self._select_action(
            self._rng_key, self._online_params, s_t, self.exploration_epsilon
        )
        a_t, v_t = jax.device_get((a_t, v_t))
        self._statistics['state_value'] = v_t
        return parts.Action(a_t)

    def _learn(self) -> None:
        
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
    def update_munchausen_epsilon(self, current_iteration: int):
        
        if current_iteration >= 30:
            step_size = 0.7 / 170
            steps_into_increase_phase = current_iteration - 30
            self.munchausen_epsilon = steps_into_increase_phase * step_size
            if self.munchausen_epsilon > 0.7:
                self.munchausen_epsilon = 0.7
        else:
            self.munchausen_epsilon = 0.0


    @property
    def online_params(self) -> parts.NetworkParams:
        
        return self._online_params

    @property
    def statistics(self) -> Mapping[Text, float]:
        
        return {k: float(v) for k, v in self._statistics.items()}

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
            'max_seen_priority': self._max_seen_priority,
        }
        return state

    def set_state(self, state: Mapping[Text, Any]) -> None:
        
        self._rng_key = state['rng_key']
        self._frame_t = state['frame_t']
        self._opt_state = jax.device_put(state['opt_state'])
        self._online_params = jax.device_put(state['online_params'])
        self._target_params = jax.device_put(state['target_params'])
        self._replay.set_state(state['replay'])
        self._max_seen_priority = state['max_seen_priority']




class IqnEpsilonGreedyActor(parts.Agent):
    

    def __init__(
        self,
        preprocessor: processors.Processor,
        network: parts.Network,
        exploration_epsilon: float,
        tau_samples: int,
        rng_key: parts.PRNGKey,
    ):
        self._preprocessor = preprocessor
        self._rng_key = rng_key
        self._action = None
        self.network_params = None

        def select_action(rng_key, network_params, s_t):
            
            rng_key, tau_key, apply_key, policy_key = jax.random.split(rng_key, 4)
            tau_t = _sample_tau(tau_key, (1, tau_samples))
            q_t = network.apply(
                network_params, apply_key, IqnInputs(s_t[None, ...], tau_t)
            ).q_values[0]
            a_t = distrax.EpsilonGreedy(q_t, exploration_epsilon).sample(
                seed=policy_key
            )
            return rng_key, a_t

        self._select_action = jax.jit(select_action)

    def step(self, timestep: dm_env.TimeStep) -> parts.Action:
        
        timestep = self._preprocessor(timestep)

        if timestep is None:  # Repeat action.
            return self._action

        s_t = timestep.observation
        self._rng_key, a_t = self._select_action(
            self._rng_key, self.network_params, s_t
        )
        self._action = parts.Action(jax.device_get(a_t))
        return self._action

    def reset(self) -> None:
        
        processors.reset(self._preprocessor)
        self._action = None

    def get_state(self) -> Mapping[Text, Any]:
        
        return {
            'rng_key': self._rng_key,
            'network_params': self.network_params,
        }

    def set_state(self, state: Mapping[Text, Any]) -> None:
        
        self._rng_key = state['rng_key']
        self.network_params = state['network_params']

    @property
    def statistics(self) -> Mapping[Text, float]:
        return {}