"""A MIQN_star agent training on Atari.

From the "Munchausen Reinforcement Learning"
*   Double Q-learning
*   Prioritized experience replay (proportional variant)
*   Dueling networks
*   Multi-step learning
*   Implicit Quantile Networks
*   Noisy networks
*   Dynamic regularization scaling
*   Munchausen regularization
"""


import collections
import itertools
import sys
import typing

from absl import app
from absl import flags
from absl import logging
import chex
import dm_env
import haiku as hk
import jax
from jax.config import config
import numpy as np
import optax

from dqn_zoo import atari_data
from dqn_zoo import gym_atari
from dqn_zoo import networks
from dqn_zoo import parts
from dqn_zoo import processors
from dqn_zoo import replay as replay_lib
from dqn_zoo.miqn_star import agent as agent

FLAGS = flags.FLAGS
_ENVIRONMENT_NAME = flags.DEFINE_string('environment_name', 'amidar', '')
_ENVIRONMENT_HEIGHT = flags.DEFINE_integer('environment_height', 84, '')
_ENVIRONMENT_WIDTH = flags.DEFINE_integer('environment_width', 84, '')
_REPLAY_CAPACITY = flags.DEFINE_integer('replay_capacity', int(1e6), '')
_COMPRESS_STATE = flags.DEFINE_bool('compress_state', True, '')
_MIN_REPLAY_CAPACITY_FRACTION = flags.DEFINE_float(
    'min_replay_capacity_fraction', 0.05, ''
)
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 32, '')
_MAX_FRAMES_PER_EPISODE = flags.DEFINE_integer(
    'max_frames_per_episode', 108000, ''
)  # 30 mins.
_NUM_ACTION_REPEATS = flags.DEFINE_integer('num_action_repeats', 4, '')
_NUM_STACKED_FRAMES = flags.DEFINE_integer('num_stacked_frames', 4, '')
_EXPLORATION_EPSILON_BEGIN_VALUE = flags.DEFINE_float(
    'exploration_epsilon_begin_value', 1.0, ''
)
_EXPLORATION_EPSILON_END_VALUE = flags.DEFINE_float(
    'exploration_epsilon_end_value', 0.01, ''
)
_EXPLORATION_EPSILON_DECAY_FRAME_FRACTION = flags.DEFINE_float(
    'exploration_epsilon_decay_frame_fraction', 0.02, ''
)
_EVAL_EXPLORATION_EPSILON = flags.DEFINE_float(
    'eval_exploration_epsilon', 0.001, ''
)
_TARGET_NETWORK_UPDATE_PERIOD = flags.DEFINE_integer(
    'target_network_update_period', int(4e4), ''
)
_TAU = flags.DEFINE_float(
    'tau', 0.03, 'Tau value used in Munchausen term calculations.'
)
_ALPHA = flags.DEFINE_float(
    'alpha', 0.9, 'Alpha value used for scaling the Munchausen term.'
)
_CLIP_VALUE_MIN = flags.DEFINE_float(
    'clip_value_min', -1.0, 'Minimum clip value for Munchausen term.'
)
_HUBER_PARAM = flags.DEFINE_float('huber_param', 1.0, '')
_LEARNING_RATE = flags.DEFINE_float('learning_rate', 5e-5, '')
_OPTIMIZER_EPSILON = flags.DEFINE_float('optimizer_epsilon', 0.0003125, '')
_ADDITIONAL_DISCOUNT = flags.DEFINE_float('additional_discount', 0.99, '')
_MAX_ABS_REWARD = flags.DEFINE_float('max_abs_reward', 1.0, '')
_MAX_GLOBAL_GRAD_NORM = flags.DEFINE_float('max_global_grad_norm', 10.0, '')
_SEED = flags.DEFINE_integer('seed', 1, '')  # GPU may introduce nondeterminism.
_NUM_ITERATIONS = flags.DEFINE_integer('num_iterations', 200, '')
_NUM_TRAIN_FRAMES = flags.DEFINE_integer(
    'num_train_frames', int(1e6), ''
)  # Per iteration.
_NUM_EVAL_FRAMES = flags.DEFINE_integer(
    'num_eval_frames', int(5e5), ''
)  # Per iteration.
_LEARN_PERIOD = flags.DEFINE_integer('learn_period', 16, '')
_RESULTS_CSV_PATH = flags.DEFINE_string(
    'results_csv_path', 'results/results_miqn_star_amidar_seed1.csv', ''
)

_PRIORITY_EXPONENT = flags.DEFINE_float('priority_exponent', 0.5, '')
_IMPORTANCE_SAMPLING_EXPONENT_BEGIN_VALUE = flags.DEFINE_float(
    'importance_sampling_exponent_begin_value', 0.4, ''
)
_IMPORTANCE_SAMPLING_EXPONENT_END_VALUE = flags.DEFINE_float(
    'importance_sampling_exponent_end_value', 1.0, ''
)
_UNIFORM_SAMPLE_PROBABILITY = flags.DEFINE_float(
    'uniform_sample_probability', 1e-3, ''
)
_NORMALIZE_WEIGHTS = flags.DEFINE_bool('normalize_weights', True, '')
_N_STEPS = flags.DEFINE_integer('n_steps', 3, '')

_TAU_LATENT_DIM = flags.DEFINE_integer('tau_latent_dim', 64, '')
_TAU_SAMPLES_POLICY = flags.DEFINE_integer('tau_samples_policy', 64, '')
_TAU_SAMPLES_S_TM1 = flags.DEFINE_integer('tau_samples_s_tm1', 64, '')
_TAU_SAMPLES_S_T = flags.DEFINE_integer('tau_samples_s_t', 64, '')
_NOISY_WEIGHT_INIT = flags.DEFINE_float('noisy_weight_init', 0.1, '')

def main(argv):
  
  del argv
  logging.info('MIQN_star on Atari on %s.', jax.lib.xla_bridge.get_backend().platform)
  random_state = np.random.RandomState(_SEED.value)
  rng_key = jax.random.PRNGKey(
      random_state.randint(-sys.maxsize - 1, sys.maxsize + 1, dtype=np.int64)
  )

  if _RESULTS_CSV_PATH.value:
    writer = parts.CsvWriter(_RESULTS_CSV_PATH.value)
  else:
    writer = parts.NullWriter()

  def environment_builder():
    
    env = gym_atari.GymAtari(
        _ENVIRONMENT_NAME.value, seed=random_state.randint(1, 2**32)
    )
    return gym_atari.RandomNoopsEnvironmentWrapper(
        env,
        min_noop_steps=1,
        max_noop_steps=30,
        seed=random_state.randint(1, 2**32),
    )

  env = environment_builder()

  logging.info('Environment: %s', _ENVIRONMENT_NAME.value)
  logging.info('Action spec: %s', env.action_spec())
  logging.info('Observation spec: %s', env.observation_spec())
  num_actions = env.action_spec().num_values
  network_fn = networks.miqn_atari_network(num_actions, _TAU_LATENT_DIM.value)
  network = hk.transform(network_fn)

  def preprocessor_builder():
    return processors.atari(
        additional_discount=_ADDITIONAL_DISCOUNT.value,
        max_abs_reward=_MAX_ABS_REWARD.value,
        resize_shape=(_ENVIRONMENT_HEIGHT.value, _ENVIRONMENT_WIDTH.value),
        num_action_repeats=_NUM_ACTION_REPEATS.value,
        num_pooled_frames=2,
        zero_discount_on_life_loss=True,
        num_stacked_frames=_NUM_STACKED_FRAMES.value,
        grayscaling=True,
    )

  sample_processed_timestep = preprocessor_builder()(env.reset())
  sample_processed_timestep = typing.cast(
      dm_env.TimeStep, sample_processed_timestep
  )
  sample_network_input = agent.IqnInputs(
      state=sample_processed_timestep.observation,
      taus=np.zeros(1, dtype=np.float32),
  )
  chex.assert_shape(
      sample_network_input.state,
      (
          _ENVIRONMENT_HEIGHT.value,
          _ENVIRONMENT_WIDTH.value,
          _NUM_STACKED_FRAMES.value,
      ),
  )

  exploration_epsilon_schedule = parts.LinearSchedule(
      begin_t=int(
          _MIN_REPLAY_CAPACITY_FRACTION.value
          * _REPLAY_CAPACITY.value
          * _NUM_ACTION_REPEATS.value
      ),
      decay_steps=int(
          _EXPLORATION_EPSILON_DECAY_FRAME_FRACTION.value
          * _NUM_ITERATIONS.value
          * _NUM_TRAIN_FRAMES.value
      ),
      begin_value=_EXPLORATION_EPSILON_BEGIN_VALUE.value,
      end_value=_EXPLORATION_EPSILON_END_VALUE.value,
  )


  importance_sampling_exponent_schedule = parts.LinearSchedule(
      begin_t=int(_MIN_REPLAY_CAPACITY_FRACTION.value * _REPLAY_CAPACITY.value),
      end_t=(
          _NUM_ITERATIONS.value
          * int(_NUM_TRAIN_FRAMES.value / _NUM_ACTION_REPEATS.value)
      ),
      begin_value=_IMPORTANCE_SAMPLING_EXPONENT_BEGIN_VALUE.value,
      end_value=_IMPORTANCE_SAMPLING_EXPONENT_END_VALUE.value,
  )

  if _COMPRESS_STATE.value:

    def encoder(transition):
      return transition._replace(
          s_tm1=replay_lib.compress_array(transition.s_tm1),
          s_t=replay_lib.compress_array(transition.s_t),
      )

    def decoder(transition):
      return transition._replace(
          s_tm1=replay_lib.uncompress_array(transition.s_tm1),
          s_t=replay_lib.uncompress_array(transition.s_t),
      )

  else:
    encoder = None
    decoder = None

  replay_structure = replay_lib.Transition(
      s_tm1=None,
      a_tm1=None,
      r_t=None,
      discount_t=None,
      s_t=None,
  )

  replay = replay_lib.TransitionReplay(
      _REPLAY_CAPACITY.value, replay_structure, random_state, encoder, decoder
  )

  transition_accumulator = replay_lib.NStepTransitionAccumulator(_N_STEPS.value)
  replay = replay_lib.PrioritizedTransitionReplay(
      _REPLAY_CAPACITY.value,
      replay_structure,
      _PRIORITY_EXPONENT.value,
      importance_sampling_exponent_schedule,
      _UNIFORM_SAMPLE_PROBABILITY.value,
      _NORMALIZE_WEIGHTS.value,
      random_state,
      encoder,
      decoder,
  )

  optimizer = optax.adam(
      learning_rate=_LEARNING_RATE.value, eps=_OPTIMIZER_EPSILON.value
  )
  if _MAX_GLOBAL_GRAD_NORM.value > 0:
    optimizer = optax.chain(
        optax.clip_by_global_norm(_MAX_GLOBAL_GRAD_NORM.value), optimizer
    )



  train_rng_key, eval_rng_key = jax.random.split(rng_key)

  train_agent = agent.miqn_star(
      preprocessor=preprocessor_builder(),
      sample_network_input=sample_network_input,
      network=network,
      optimizer=optimizer,
      transition_accumulator=transition_accumulator,
      replay=replay,
      batch_size=_BATCH_SIZE.value,
      exploration_epsilon=exploration_epsilon_schedule,
      min_replay_capacity_fraction=_MIN_REPLAY_CAPACITY_FRACTION.value,
      learn_period=_LEARN_PERIOD.value,
      target_network_update_period=_TARGET_NETWORK_UPDATE_PERIOD.value,
      huber_param=_HUBER_PARAM.value,
      tau_samples_policy=_TAU_SAMPLES_POLICY.value,
      tau_samples_s_tm1=_TAU_SAMPLES_S_TM1.value,
      tau_samples_s_t=_TAU_SAMPLES_S_T.value,
      rng_key=train_rng_key,
  )
  eval_agent = agent.IqnEpsilonGreedyActor(
      preprocessor=preprocessor_builder(),
      network=network,
      exploration_epsilon=_EVAL_EXPLORATION_EPSILON.value,
      tau_samples=_TAU_SAMPLES_POLICY.value,
      rng_key=eval_rng_key,
  )

  checkpoint = parts.NullCheckpoint()

  state = checkpoint.state
  state.iteration = 0
  state.train_agent = train_agent
  state.eval_agent = eval_agent
  state.random_state = random_state
  state.writer = writer
  if checkpoint.can_be_restored():
    checkpoint.restore()

  while state.iteration <= _NUM_ITERATIONS.value:
    env = environment_builder()

    logging.info('Training iteration %d.', state.iteration)
    train_seq = parts.run_loop(train_agent, env, _MAX_FRAMES_PER_EPISODE.value)
    num_train_frames = 0 if state.iteration == 0 else _NUM_TRAIN_FRAMES.value
    train_seq_truncated = itertools.islice(train_seq, num_train_frames)
    train_trackers = parts.make_default_trackers(train_agent)
    train_stats = parts.generate_statistics(train_trackers, train_seq_truncated)
    train_agent.update_munchausen_epsilon(state.iteration)
    logging.info('Muchausen Esilon %f.', train_agent.munchausen_epsilon)
    logging.info('Evaluation iteration %d.', state.iteration)
    eval_agent.network_params = train_agent.online_params
    eval_seq = parts.run_loop(eval_agent, env, _MAX_FRAMES_PER_EPISODE.value)
    eval_seq_truncated = itertools.islice(eval_seq, _NUM_EVAL_FRAMES.value)
    eval_trackers = parts.make_default_trackers(eval_agent)
    eval_stats = parts.generate_statistics(eval_trackers, eval_seq_truncated)

    human_normalized_score = atari_data.get_human_normalized_score(
        _ENVIRONMENT_NAME.value, eval_stats['episode_return']
    )
    capped_human_normalized_score = np.amin([1.0, human_normalized_score])
    log_output = [
        ('iteration', state.iteration, '%3d'),
        ('frame', state.iteration * _NUM_TRAIN_FRAMES.value, '%5d'),
        ('eval_episode_return', eval_stats['episode_return'], '% 2.2f'),
        ('train_episode_return', train_stats['episode_return'], '% 2.2f'),
        ('eval_num_episodes', eval_stats['num_episodes'], '%3d'),
        ('train_num_episodes', train_stats['num_episodes'], '%3d'),
        ('eval_frame_rate', eval_stats['step_rate'], '%4.0f'),
        ('train_frame_rate', train_stats['step_rate'], '%4.0f'),
        ('train_exploration_epsilon', train_agent.exploration_epsilon, '%.3f'),
        ('train_state_value', train_stats['state_value'], '%.3f'),
        ('normalized_return', human_normalized_score, '%.3f'),
        ('capped_normalized_return', capped_human_normalized_score, '%.3f'),
        ('human_gap', 1.0 - capped_human_normalized_score, '%.3f'),
    ]
    log_output_str = ', '.join(('%s: ' + f) % (n, v) for n, v, f in log_output)
    logging.info(log_output_str)
    writer.write(collections.OrderedDict((n, v) for n, v, _ in log_output))
    state.iteration += 1
    checkpoint.save()

  writer.close()


if __name__ == '__main__':
    import time

    start_time = time.time()
    config.update('jax_platform_name', 'gpu')  # Default to GPU.
    config.update('jax_numpy_rank_promotion', 'raise')
    config.config_with_absl()
    app.run(main)

    total_time = time.time() - start_time
    total_frames = _NUM_ITERATIONS.value * _NUM_TRAIN_FRAMES.value
    fps = total_frames / total_time

    print(f'Total running time: {total_time:.2f} seconds')
    print(f'Total frames: {total_frames:,}')
    print(f'Average speed: {fps:.2f} frames/second')
    print(f'Hours taken: {total_time / 3600:.2f}')