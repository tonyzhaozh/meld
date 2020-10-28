from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import gin
import numpy as np
import six

from tf_agents.metrics import py_metric
from tf_agents.utils import nest_utils
from tf_agents.utils import numpy_storage

from tf_agents.metrics import py_metrics


@gin.configurable
class AverageVelocityErrorMetric(py_metrics.StreamingMetric):
  """Computes the average undiscounted reward."""

  def __init__(self, name='AverageReturn', buffer_size=10, batch_size=None):
    """Creates an AverageReturnMetric."""
    self._np_state = numpy_storage.NumpyState()
    # Set a dummy value on self._np_state.vel_error so it gets included in
    # the first checkpoint (before metric is first called).
    self._np_state.vel_error = np.float64(0)
    super(AverageVelocityErrorMetric, self).__init__(name, buffer_size=buffer_size,
                                              batch_size=batch_size)

  def _reset(self, batch_size):
    """Resets stat gathering variables."""
    self._np_state.vel_error = np.zeros(
        shape=(batch_size,), dtype=np.float64)

  def _batched_call(self, trajectory):
    """Processes the trajectory to update the metric.

    Args:
      trajectory: a tf_agents.trajectory.Trajectory.
    """

    vel_error = self._np_state.vel_error

    is_first = np.where(trajectory.is_first())
    is_last = np.where(trajectory.is_last())

    vel_error[is_first] = 0
    vel_error += trajectory.observation['env_info'][:, 3]

    self.add_to_buffer(vel_error[is_last])




@gin.configurable
class AverageScoreMetric(py_metrics.StreamingMetric):
  """Computes the average undiscounted reward."""

  def __init__(self, name='AverageScore', buffer_size=10, batch_size=None):
    """Creates an AverageReturnMetric."""
    self._np_state = numpy_storage.NumpyState()
    # Set a dummy value on self._np_state.vel_error so it gets included in
    # the first checkpoint (before metric is first called).
    self._np_state.score = np.float64(0)
    super(AverageScoreMetric, self).__init__(name, buffer_size=buffer_size,
                                              batch_size=batch_size)

  def _reset(self, batch_size):
    """Resets stat gathering variables."""
    self._np_state.score = np.zeros(shape=(batch_size,), dtype=np.float64)

  def _batched_call(self, trajectory):
    """Processes the trajectory to update the metric.

    Args:
      trajectory: a tf_agents.trajectory.Trajectory.
    """

    is_first = np.where(trajectory.is_first())
    is_last = np.where(trajectory.is_last())

    self._np_state.score[is_first] = 0
    # self._np_state.score += trajectory.observation['env_info'][:, 0] ### SCORE as sum over rollout
    self._np_state.score = trajectory.observation['env_info'][:, 0] ### SCORE AT the last time

    self.add_to_buffer(self._np_state.score[is_last])



@gin.configurable
class AverageControlCostMetric(py_metrics.StreamingMetric):
  """Computes the average undiscounted reward."""

  def __init__(self, name='AverageReturn', buffer_size=10, batch_size=None):
    """Creates an AverageReturnMetric."""
    self._np_state = numpy_storage.NumpyState()
    # Set a dummy value on self._np_state.control_cost so it gets included in
    # the first checkpoint (before metric is first called).
    self._np_state.control_cost = np.float64(0)
    super(AverageControlCostMetric, self).__init__(name, buffer_size=buffer_size,
                                              batch_size=batch_size)

  def _reset(self, batch_size):
    """Resets stat gathering variables."""
    self._np_state.control_cost = np.zeros(
        shape=(batch_size,), dtype=np.float64)

  def _batched_call(self, trajectory):
    """Processes the trajectory to update the metric.

    Args:
      trajectory: a tf_agents.trajectory.Trajectory.
    """

    control_cost = self._np_state.control_cost

    is_first = np.where(trajectory.is_first())
    is_last = np.where(trajectory.is_last())

    control_cost[is_first] = 0
    control_cost += trajectory.observation['env_info'][:, 1]

    self.add_to_buffer(control_cost[is_last])