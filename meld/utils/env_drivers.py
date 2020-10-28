from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import numpy as np

from tf_agents.bandits.environments import bandit_py_environment
from tf_agents.bandits.environments import bandit_tf_environment
from tf_agents.drivers import driver
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.utils import nest_utils
from tf_agents.trajectories.time_step import TimeStep, StepType


def is_bandit_env(env):
  actual_env = env
  if isinstance(env, tf_py_environment.TFPyEnvironment):
    actual_env = env.pyenv
  is_bandit = (
      isinstance(actual_env, bandit_py_environment.BanditPyEnvironment) or
      isinstance(actual_env, bandit_tf_environment.BanditTFEnvironment))
  return is_bandit


@gin.configurable
class DynamicTrialDriver(driver.Driver):

  def __init__(self,
               env,
               policy,
               num_trials_to_collect=None,
               observers=None,
               transition_observers=None,
               episodes_per_trial=None,
               max_episode_len=None):
    """
    Creates a DynamicEpisodeDriver.

    Args:
      env: A tf_environment.Base environment.
      policy: A tf_policy.Base policy.
      observers: A list of observers that are updated after every step in the
        environment. Each observer is a callable(Trajectory).
      transition_observers: A list of observers that are updated after every
        step in the environment. Each observer is a callable((TimeStep,
        PolicyStep, NextTimeStep)).
      num_trials_to_collect: The number of trials to collect in the environment.
      episodes_per_trial: Number of episodes per trial

    Note: policy_state's internal latent is reset to 0 at beginning of each trial

    """
    super(DynamicTrialDriver, self).__init__(env, policy, observers,
                                               transition_observers)

    self.num_trials_to_collect = num_trials_to_collect
    self.episodes_per_trial = episodes_per_trial
    self.max_episode_len = max_episode_len

    self._num_episodes = self.num_trials_to_collect * self.episodes_per_trial
    self._run_fn = common.function_in_tf1()(self._run)

    self.override_step_type_fn = common.function_in_tf1()(self._override_step_type)

    self._is_bandit_env = is_bandit_env(env)

  """
  MELD custom function:
    (a) when driver's counter%episodes_per_trial==0,
      want the policy_state to reset its latent
      so pass time_step as it is
    (b) else,
      don't want policy_state to reset its latent
      so artificially change the timestep's step type from FIRST --> MID
  """
  def _override_step_type(self, time_step, counter):
    policy_time_step = tf.cond(tf.equal(tf.math.floormod(counter[0], self.episodes_per_trial), 0), # if first episode in a trial
                            lambda: time_step,
                            lambda: TimeStep(step_type=np.expand_dims(StepType.MID, axis=0),
                                             reward=time_step.reward,
                                             discount=time_step.discount,
                                             observation=time_step.observation,),)
    return policy_time_step

  def _loop_condition_fn(self, num_episodes):
    """Returns a function with the condition needed for tf.while_loop."""

    def loop_cond(counter, *_):
      """Determines when to stop the loop, based on episode counter.
      Args:
        counter: Episode counters per batch index. 
          Shape [batch_size] when batch_size > 1, else shape [].
      Returns:
        tf.bool tensor, shape (), indicating whether while loop should continue.
      """
      return tf.less(tf.reduce_sum(input_tensor=counter), num_episodes)

    return loop_cond

  def _loop_body_fn(self):
    """Returns a function with the driver's loop body ops."""

    def loop_body(counter, time_step, policy_state):
      """Runs a step in environment.
      While loop will call multiple times.
      Args:
        counter: Episode counters per batch index. Shape [batch_size].
        time_step: TimeStep tuple with elements shape [batch_size, ...].
        policy_state: Policy state tensor shape [batch_size, policy_state_dim].
          Pass empty tuple for non-recurrent policies.
      Returns:
        loop_vars for next iteration of tf.while_loop.
      """
      policy_time_step = self.override_step_type_fn(time_step, counter)
      action_step = self.policy.action(policy_time_step, policy_state)

      # TODO(b/134487572): TF2 while_loop seems to either ignore
      # parallel_iterations or doesn't properly propagate control dependencies
      # from one step to the next. Without this dep, self.env.step() is called
      # in parallel.
      with tf.control_dependencies(tf.nest.flatten([time_step])):
        next_time_step = self.env.step(action_step.action)

      policy_state = action_step.state

      if self._is_bandit_env:
        # For Bandits we create episodes of length 1.
        # Since the `next_time_step` is always of type LAST we need to replace
        # the step type of the current `time_step` to FIRST.
        batch_size = tf.shape(input=time_step.discount)
        time_step = time_step._replace(
            step_type=tf.fill(batch_size, ts.StepType.FIRST))

      traj = trajectory.from_transition(time_step, action_step, next_time_step)
      observer_ops = [observer(traj) for observer in self._observers]
      transition_observer_ops = [
          observer((time_step, action_step, next_time_step))
          for observer in self._transition_observers
      ]
      with tf.control_dependencies(
          [tf.group(observer_ops + transition_observer_ops)]):
        time_step, next_time_step, policy_state = tf.nest.map_structure(
            tf.identity, (time_step, next_time_step, policy_state))

      # While loop counter is only incremented for episode reset episodes.
      # For Bandits, this is every trajectory, for MDPs, this is at boundaries.
      if self._is_bandit_env:
        counter += tf.ones(batch_size, dtype=tf.int32)
      else:
        counter += tf.cast(traj.is_boundary(), dtype=tf.int32)

      return [counter, next_time_step, policy_state]

    return loop_body

  def run(self,
          time_step=None,
          policy_state=None,
          num_episodes=None,
          maximum_iterations=None):

    """Takes episodes in the environment using the policy and update observers.
    If `time_step` and `policy_state` are not provided, `run` will reset the
    environment and request an initial state from the policy.

    Args:
      time_step: optional initial time_step. If None, it will be obtained by
        resetting the environment. Elements should be shape [batch_size, ...].
      policy_state: optional initial state for the policy. If None, it will be
        obtained from the policy.get_initial_state().
      num_episodes: Optional number of episodes to take in the environment. If
        None it would use initial num_episodes.
      maximum_iterations: Optional maximum number of iterations of the while
        loop to run. If provided, the cond output is AND-ed with an additional
        condition ensuring the number of iterations executed is no greater than
        maximum_iterations.

    Returns:
      time_step: TimeStep named tuple with final observation, reward, etc.
      policy_state: Tensor with final step policy state.
    """
    return self._run_fn(
        time_step=time_step,
        policy_state=policy_state,
        num_episodes=num_episodes,
        maximum_iterations=maximum_iterations)

  def _run(self,
           time_step=None,
           policy_state=None,
           num_episodes=None,
           maximum_iterations=None):

    assert time_step is None
    assert policy_state is None # just for MELD, sanity check, will make sure batch_dims=1
    assert num_episodes is None

    if time_step is None:
      time_step = self.env.reset()

    if policy_state is None:
      policy_state = self.policy.get_initial_state(self.env.batch_size)

    # Batch dim should be first index of tensors during data collection.
    batch_dims = nest_utils.get_outer_shape(time_step, self.env.time_step_spec())
    counter = tf.zeros(batch_dims, tf.int32)

    num_episodes = num_episodes or self._num_episodes
    [_, time_step, policy_state] = tf.nest.map_structure(
        tf.stop_gradient,
        tf.while_loop(
            cond=self._loop_condition_fn(num_episodes),
            body=self._loop_body_fn(),
            loop_vars=[counter, time_step, policy_state],
            parallel_iterations=1,
            maximum_iterations=maximum_iterations,
            name='driver_loop'))

    return time_step, policy_state