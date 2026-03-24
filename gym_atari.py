
from typing import Optional, Tuple

import atari_py  # pylint: disable=unused-import for gym to load Atari games.
import dm_env
from dm_env import specs
import gym
import numpy as np

from dqn_zoo import atari_data

_GYM_ID_SUFFIX = '-xitari-v1'


def _register_atari_environments():
  
  for game in atari_data.ATARI_GAMES:
    gym.envs.registration.register(
        id=game + _GYM_ID_SUFFIX,  # Add suffix so ID has required format.
        entry_point='gym.envs.atari:AtariEnv',
        kwargs={  # Explicitly set all known arguments.
            'game': game,
            'mode': None,  # Not necessarily the same as 0.
            'difficulty': None,  # Not necessarily the same as 0.
            'obs_type': 'image',
            'frameskip': 1,  # Get every frame.
            'repeat_action_probability': 0.0,  # No sticky actions.
            'full_action_space': False,
        },
        max_episode_steps=None,  # No time limit, handled in training run loop.
        nondeterministic=False,  # Xitari is deterministic.
    )


_register_atari_environments()


class GymAtari(dm_env.Environment):
  

  def __init__(self, game, seed):
    self._gym_env = gym.make(game + _GYM_ID_SUFFIX)
    self._gym_env.seed(seed)
    self._start_of_episode = True

  def reset(self) -> dm_env.TimeStep:
    
    observation = self._gym_env.reset()
    lives = np.int32(self._gym_env.ale.lives())
    timestep = dm_env.restart((observation, lives))
    self._start_of_episode = False
    return timestep

  def step(self, action: np.int32) -> dm_env.TimeStep:
    
    if self._start_of_episode:
      step_type = dm_env.StepType.FIRST
      observation = self._gym_env.reset()
      discount = None
      reward = None
      done = False
    else:
      observation, reward, done, info = self._gym_env.step(action)
      if done:
        assert 'TimeLimit.truncated' not in info, 'Should never truncate.'
        step_type = dm_env.StepType.LAST
        discount = 0.0
      else:
        step_type = dm_env.StepType.MID
        discount = 1.0

    lives = np.int32(self._gym_env.ale.lives())
    timestep = dm_env.TimeStep(
        step_type=step_type,
        observation=(observation, lives),
        reward=reward,
        discount=discount,
    )
    self._start_of_episode = done
    return timestep

  def observation_spec(self) -> Tuple[specs.Array, specs.Array]:
    space = self._gym_env.observation_space
    return (
        specs.Array(shape=space.shape, dtype=space.dtype, name='rgb'),
        specs.Array(shape=(), dtype=np.int32, name='lives'),
    )

  def action_spec(self) -> specs.DiscreteArray:
    space = self._gym_env.action_space
    return specs.DiscreteArray(
        num_values=space.n, dtype=np.int32, name='action'
    )

  def close(self):
    self._gym_env.close()


class RandomNoopsEnvironmentWrapper(dm_env.Environment):
  

  def __init__(
      self,
      environment: dm_env.Environment,
      max_noop_steps: int,
      min_noop_steps: int = 0,
      noop_action: int = 0,
      seed: Optional[int] = None,
  ):
    
    self._environment = environment
    if max_noop_steps < min_noop_steps:
      raise ValueError('max_noop_steps must be greater or equal min_noop_steps')
    self._min_noop_steps = min_noop_steps
    self._max_noop_steps = max_noop_steps
    self._noop_action = noop_action
    self._rng = np.random.RandomState(seed)

  def reset(self):
    
    return self._apply_random_noops(initial_timestep=self._environment.reset())

  def step(self, action):
    
    timestep = self._environment.step(action)
    if timestep.first():
      return self._apply_random_noops(initial_timestep=timestep)
    else:
      return timestep

  def _apply_random_noops(self, initial_timestep):
    assert initial_timestep.first()
    num_steps = self._rng.randint(
        self._min_noop_steps, self._max_noop_steps + 1
    )
    timestep = initial_timestep
    for _ in range(num_steps):
      timestep = self._environment.step(self._noop_action)
      if timestep.last():
        raise RuntimeError(
            'Episode ended while applying %s noop actions.' % num_steps
        )

    return dm_env.restart(timestep.observation)


  def observation_spec(self):
    return self._environment.observation_spec()

  def action_spec(self):
    return self._environment.action_spec()

  def reward_spec(self):
    return self._environment.reward_spec()

  def discount_spec(self):
    return self._environment.discount_spec()

  def close(self):
    return self._environment.close()