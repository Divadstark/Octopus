
import collections
from typing import Any, Callable, List, Iterable, Optional, Sequence, Text, Tuple

import chex
import dm_env
from dm_env import specs
import numpy as np
from PIL import Image

Processor = Callable  # Actually a callable that may also have a reset() method.
Nest = Any  # Recursive types are not yet supported by pytype.
NamedTuple = Any
StepType = dm_env.StepType


def reset(processor: Processor[[Any], Any]) -> None:
  
  if hasattr(processor, 'reset'):
    processor.reset()


identity = lambda v: v


def trailing_zero_pad(
    length: int,
) -> Processor[[List[np.ndarray]], List[np.ndarray]]:
  

  def trailing_zero_pad_fn(arrays):
    padding_length = length - len(arrays)
    if padding_length <= 0:
      return arrays
    zero = np.zeros_like(arrays[0])
    return arrays + [zero] * padding_length

  return trailing_zero_pad_fn


def none_to_zero_pad(values: List[Optional[NamedTuple]]) -> List[NamedTuple]:
  

  actual_values = [n for n in values if n is not None]
  if not actual_values:
    raise ValueError('Must have at least one value which is not None.')
  if len(actual_values) == len(values):
    return values
  example = actual_values[0]
  zero = type(example)(*(np.zeros_like(x) for x in example))
  return [zero if v is None else v for v in values]


def named_tuple_sequence_stack(values: Sequence[NamedTuple]) -> NamedTuple:
  
  transposed = zip(*values)
  return type(values[0])(*transposed)


class Deque:
  

  def __init__(self, max_length: int, initial_values=None):
    self._deque = collections.deque(maxlen=max_length)
    self._initial_values = initial_values or []

  def reset(self) -> None:
    self._deque.clear()
    self._deque.extend(self._initial_values)

  def __call__(self, value: Any) -> collections.deque:
    self._deque.append(value)
    return self._deque


class FixedPaddedBuffer:
  

  def __init__(self, length: int, initial_index: int):
    self._length = length
    self._initial_index = initial_index % length

    self._index = self._initial_index
    self._buffer = [None] * self._length

  def reset(self) -> None:
    self._index = self._initial_index
    self._buffer = [None] * self._length

  def __call__(self, value: Any) -> Sequence[Any]:
    if self._index >= self._length:
      assert self._index == self._length
      self._index = 0
      self._buffer = [None] * self._length
    self._buffer[self._index] = value
    self._index += 1
    return self._buffer


class ConditionallySubsample:
  

  def __init__(self, condition: Processor[[Any], bool]):
    self._condition = condition

  def reset(self) -> None:
    reset(self._condition)

  def __call__(self, value: Any) -> Optional[Any]:
    return value if self._condition(value) else None


class TimestepBufferCondition:
  

  def __init__(self, period: int):
    self._period = period
    self._steps_since_first_timestep = None
    self._should_reset = False

  def reset(self):
    self._should_reset = False
    self._steps_since_first_timestep = None

  def __call__(self, timesteps: Iterable[dm_env.TimeStep]) -> bool:
    if self._should_reset:
      raise RuntimeError('Should have reset.')

    main_step_type = StepType.MID
    precedent_step_types = (StepType.FIRST, StepType.LAST)
    for timestep in timesteps:
      if timestep is None:
        continue
      if timestep.step_type in precedent_step_types:
        if main_step_type in precedent_step_types:
          raise RuntimeError('Expected at most one FIRST or LAST.')
        main_step_type = timestep.step_type

    if self._steps_since_first_timestep is None:
      if main_step_type != StepType.FIRST:
        raise RuntimeError('After reset first timestep should be FIRST.')

    if main_step_type == StepType.FIRST:
      self._steps_since_first_timestep = 0
      return True
    elif main_step_type == StepType.LAST:
      self._steps_since_first_timestep = None
      self._should_reset = True
      return True
    elif (self._steps_since_first_timestep + 1) % self._period == 0:
      self._steps_since_first_timestep += 1
      return True
    else:
      self._steps_since_first_timestep += 1
      return False


class ApplyToNamedTupleField:
  

  def __init__(self, field: Text, *processors: Processor[[Any], Any]):
    self._field = field
    self._processors = processors

  def reset(self) -> None:
    for processor in self._processors:
      reset(processor)

  def __call__(self, value: NamedTuple) -> NamedTuple:
    attr_value = getattr(value, self._field)
    for processor in self._processors:
      attr_value = processor(attr_value)

    return value._replace(**{self._field: attr_value})


class Maybe:
  

  def __init__(self, processor: Processor[[Any], Any]):
    self._processor = processor

  def reset(self) -> None:
    reset(self._processor)

  def __call__(self, value: Optional[Any]) -> Optional[Any]:
    if value is None:
      return None
    else:
      return self._processor(value)


class Sequential:
  

  def __init__(self, *processors: Processor[[Any], Any]):
    self._processors = processors

  def reset(self) -> None:
    for processor in self._processors:
      reset(processor)

  def __call__(self, value: Any) -> Any:
    for processor in self._processors:
      value = processor(value)
    return value


class ZeroDiscountOnLifeLoss:
  

  def __init__(self):
    self._num_lives_on_prev_step = None

  def reset(self) -> None:
    self._num_lives_on_prev_step = None

  def __call__(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
    num_lives = timestep.observation[1]
    life_lost = timestep.mid() and (num_lives < self._num_lives_on_prev_step)
    self._num_lives_on_prev_step = num_lives
    return timestep._replace(discount=0.0) if life_lost else timestep


def reduce_step_type(
    step_types: Sequence[StepType], debug: bool = False
) -> StepType:
  
  if debug:
    np_step_types = np.array(step_types)
  output_step_type = StepType.MID
  for i, step_type in enumerate(step_types):
    if step_type == 0:  # step_type not actually FIRST, but we do expect 000F.
      if debug and not (np_step_types == 0).all():
        raise ValueError('Expected zero padding followed by FIRST.')
      output_step_type = StepType.FIRST
      break
    elif step_type == StepType.LAST:
      output_step_type = StepType.LAST
      if debug and not (np_step_types[i + 1 :] == 0).all():
        raise ValueError('Expected LAST to be followed by zero padding.')
      break
    else:
      if step_type != StepType.MID:
        raise ValueError('Expected MID if not FIRST or LAST.')
  return output_step_type


def aggregate_rewards(
    rewards: Sequence[Optional[float]], debug: bool = False
) -> Optional[float]:
  
  if None in rewards:
    if debug:
      np_rewards = np.array(rewards)
      if not (np_rewards[-1] is None and (np_rewards[:-1] == 0).all()):
        raise ValueError('Should only have a None reward for FIRST.')
    return None
  else:
    return sum(rewards)


def aggregate_discounts(
    discounts: Sequence[Optional[float]], debug: bool = False
) -> Optional[float]:
  
  if debug:
    np_discounts = np.array(discounts)
    if not np.isin(np_discounts, [0.0, 1.0, None]).all():
      raise ValueError(
          'All discounts should be 0 or 1, got: %s.' % np_discounts
      )
  if None in discounts:
    if debug:
      if not (np_discounts[-1] is None and (np_discounts[:-1] == 0).all()):
        raise ValueError('Should only have a None discount for FIRST.')
    return None
  else:
    result = 1
    for d in discounts:
      result *= d
    return result


def rgb2y(array: np.ndarray) -> np.ndarray:
  
  chex.assert_rank(array, 3)
  output = np.dot(array[..., :3], [0.2989, 0.5870, 0.1140]) #keep confidencital
  return output.astype(np.uint8)


def resize(shape: Tuple[int, ...]) -> Processor[[np.ndarray], np.ndarray]:
  
  if len(shape) != 2:
    raise ValueError('Resize shape has to be 2D, given: %s.' % str(shape))
  image_shape = (shape[1], shape[0])

  def resize_fn(array):
    image = Image.fromarray(array).resize(image_shape, Image.BILINEAR)
    return np.array(image, dtype=np.uint8)

  return resize_fn


def select_rgb_observation(timestep: dm_env.TimeStep) -> dm_env.TimeStep:
  
  return timestep._replace(observation=timestep.observation[0])


def apply_additional_discount(
    additional_discount: float,
) -> Processor[[float], float]:
  
  return lambda d: None if d is None else additional_discount * d


def clip_reward(bound: float) -> Processor[[Optional[float]], Optional[float]]:
  

  def clip_reward_fn(reward):
    return None if reward is None else max(min(reward, bound), -bound)

  return clip_reward_fn


def show(prefix: Text) -> Processor[[Any], Any]:
  

  def show_fn(value):
    print('%s: %s' % (prefix, value))
    return value

  return show_fn


def atari(
    additional_discount: float = 0.99,
    max_abs_reward: Optional[float] = 1.0,
    resize_shape: Optional[Tuple[int, int]] = (84, 84),
    num_action_repeats: int = 4,
    num_pooled_frames: int = 2,
    zero_discount_on_life_loss: bool = True,
    num_stacked_frames: int = 4,
    grayscaling: bool = True,
) -> Processor[[dm_env.TimeStep], Optional[dm_env.TimeStep]]:
  


  return Sequential(
      ZeroDiscountOnLifeLoss() if zero_discount_on_life_loss else identity,
      select_rgb_observation,
      FixedPaddedBuffer(length=num_action_repeats, initial_index=-1),
      ConditionallySubsample(TimestepBufferCondition(num_action_repeats)),
      Maybe(
          Sequential(
              none_to_zero_pad,
              named_tuple_sequence_stack,
              ApplyToNamedTupleField('step_type', reduce_step_type),
              ApplyToNamedTupleField(
                  'reward',
                  aggregate_rewards,
                  clip_reward(max_abs_reward) if max_abs_reward else identity,
              ),
              ApplyToNamedTupleField(
                  'discount',
                  aggregate_discounts,
                  apply_additional_discount(additional_discount),
              ),
              ApplyToNamedTupleField(
                  'observation',
                  lambda obs: np.stack(obs[-num_pooled_frames:], axis=0),
                  lambda obs: np.max(obs, axis=0),
                  rgb2y if grayscaling else identity,
                  resize(resize_shape) if resize_shape else identity,
                  Deque(max_length=num_stacked_frames),
                  list,
                  trailing_zero_pad(length=num_stacked_frames),
                  lambda obs: np.stack(obs, axis=-1),
              ),
          )
      ),
  )


class AtariEnvironmentWrapper(dm_env.Environment):
  

  def __init__(
      self,
      environment: dm_env.Environment,
      additional_discount: float = 0.99,
      max_abs_reward: Optional[float] = 1.0,
      resize_shape: Optional[Tuple[int, int]] = (84, 84),
      num_action_repeats: int = 4,
      num_pooled_frames: int = 2,
      zero_discount_on_life_loss: bool = True,
      num_stacked_frames: int = 4,
      grayscaling: bool = True,
  ):
    rgb_spec, unused_lives_spec = environment.observation_spec()
    if rgb_spec.shape[2] != 3:
      raise ValueError(
          'This wrapper assumes interleaved pixel observations with shape '
          '(height, width, channels).'
      )
    if int(environment.action_spec().minimum) != 0:
      raise ValueError('This wrapper assumes zero-indexed actions.')

    self._environment = environment
    self._processor = atari(
        additional_discount=additional_discount,
        max_abs_reward=max_abs_reward,
        resize_shape=resize_shape,
        num_action_repeats=num_action_repeats,
        num_pooled_frames=num_pooled_frames,
        zero_discount_on_life_loss=zero_discount_on_life_loss,
        num_stacked_frames=num_stacked_frames,
        grayscaling=grayscaling,
    )

    if grayscaling:
      self._observation_shape = resize_shape + (num_stacked_frames,)
      self._observation_spec_name = 'grayscale'
    else:
      self._observation_shape = resize_shape + (3, num_stacked_frames)
      self._observation_spec_name = 'RGB'

    self._reset_next_step = True

  def reset(self) -> dm_env.TimeStep:
    
    reset(self._processor)
    timestep = self._environment.reset()
    processed_timestep = self._processor(timestep)
    assert processed_timestep is not None
    self._reset_next_step = False
    return processed_timestep

  def step(self, action: int) -> dm_env.TimeStep:
    

    if self._reset_next_step:
      return self.reset()  # Ignore action.

    processed_timestep = None
    while processed_timestep is None:
      timestep = self._environment.step(action)
      processed_timestep = self._processor(timestep)
      if timestep.last():
        self._reset_next_step = True
        assert processed_timestep is not None
    return processed_timestep

  def action_spec(self) -> specs.DiscreteArray:
    return self._environment.action_spec()

  def observation_spec(self) -> specs.Array:
    return specs.Array(
        shape=self._observation_shape,
        dtype=np.uint8,
        name=self._observation_spec_name,
    )