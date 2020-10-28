from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import gym
import gym.spaces
import numpy as np
from tf_agents.environments import wrappers
import numpy as np

import gin

@gin.configurable
class RenderGymWrapper(wrappers.PyEnvironmentBaseWrapper):

  def __init__(self, gym_env, render_kwargs=None, goal_visibile_only_in_videos=False):
    super(RenderGymWrapper, self).__init__(gym_env)
    self._render_kwargs = dict(
        width=64,
        height=64,
        depth=False,
        camera_name='track',
    )
    if render_kwargs is not None:
      self._render_kwargs.update(render_kwargs)

    self.goal_visibile_only_in_videos = goal_visibile_only_in_videos

  @property
  def sim(self):
    return self._env.sim

  def render(self, mode='rgb_array'):
    if mode == 'rgb_array':
      if self.goal_visibile_only_in_videos:
        self._env.goal_visibility(True)
        ret = self._env.sim.render(**self._render_kwargs)[::-1, :, :]
        self._env.goal_visibility(False)
        return ret
      else:
        return self._env.sim.render(**self._render_kwargs)[::-1, :, :]

    else:
      return self._env.render(mode=mode)

@gin.configurable
class PixelObservationsGymWrapper(wrappers.PyEnvironmentBaseWrapper):

  def __init__(self, gym_env, observations_whitelist=None, render_kwargs=None, auxCamSettings=None):

    super(PixelObservationsGymWrapper, self).__init__(gym_env)
    self.max_size_of_envinfo = 5  # parameter: observation_spaces['env_info'] size
    if observations_whitelist is None:
      self._observations_whitelist = ['state', 'pixels', "env_info"]
    else:
      self._observations_whitelist = observations_whitelist

    self._render_kwargs_aux = auxCamSettings
    if self._render_kwargs_aux is not None:
      self._has_aux_cam = True
      self._observations_whitelist.append("aux_camera")
    else:
      self._has_aux_cam = False

    self._render_kwargs = dict(
        width=64,
        height=64,
        depth=False,
        camera_name='track',
    ) # default
    if render_kwargs is not None:
      self._render_kwargs.update(render_kwargs) # update default

    if self._render_kwargs_aux is not None:
      # some sanity check
      assert self._render_kwargs_aux['width'] == self._render_kwargs['width']
      assert self._render_kwargs_aux['height'] == self._render_kwargs['height']

    observation_spaces = collections.OrderedDict()
    for observation_name in self._observations_whitelist:
      if observation_name == 'state':
        observation_spaces['state'] = self._env.observation_space
      elif observation_name == 'pixels':
        if self._has_aux_cam:
          image_shape = (
            self._render_kwargs['width']*2, self._render_kwargs['height'], 3)
        else:
          image_shape = (
            self._render_kwargs['width'], self._render_kwargs['height'], 3)
        image_space = gym.spaces.Box(0, 255, shape=image_shape, dtype=np.uint8)
        observation_spaces['pixels'] = image_space
      elif observation_name == "env_info":
        observation_spaces['env_info'] = gym.spaces.Box(-100*np.ones(5), 100*np.ones(self.max_size_of_envinfo))
      ### Two cameras setting
      elif observation_name == "aux_camera":
        image_shape = (
          self._render_kwargs_aux['width'], self._render_kwargs_aux['height'], 3)
        image_space = gym.spaces.Box(0, 255, shape=image_shape, dtype=np.uint8)
        observation_spaces['aux_camera'] = image_space
      else:
        raise ValueError('observations_whitelist can only have "state" '
                         'or "pixels" or "env_info", got %s.' % observation_name)
    self.observation_space = gym.spaces.Dict(observation_spaces)

  def _modify_observation(self, observation, info=None):

    if info is None:
      zero_list = [0]*self.max_size_of_envinfo
      info = np.array(zero_list)

    observations = collections.OrderedDict()
    for observation_name in self._observations_whitelist:
      if observation_name == 'state':
        observations['state'] = observation
      elif observation_name == 'pixels':
        if self._has_aux_cam: # set both observations
          image = self._env.sim.render(**self._render_kwargs)[::-1, :, :]
          image_aux_camera = self._env.sim.render(**self._render_kwargs_aux)[::-1, :, :]
          image = np.concatenate((image, image_aux_camera), axis=0)
          observations['aux_camera'] = np.swapaxes(np.array(image_aux_camera, copy=True), 0, 1)
          observations['pixels'] = image
        else:  # set only pixel observation
          image = self._env.sim.render(**self._render_kwargs)[::-1, :, :]
          observations['pixels'] = image
      elif observation_name == "env_info":
        observations['env_info'] = info
      elif observation_name == "aux_camera":
        assert self._render_kwargs_aux
      else:
        raise ValueError('observations_whitelist can only have "state" '
                         'or "pixels" or "env_info", got %s.' % observation_name)
    return observations

  def _step(self, action):
    observation, reward, done, info = self._env.step(action)
    observation = self._modify_observation(observation, info)
    return observation, reward, done, info

  def _reset(self):
    observation = self._env.reset()
    return self._modify_observation(observation)

  def render(self, mode='rgb_array'):
    return self._env.render(mode=mode)


@gin.configurable
class PixelObservationsGymWrapperState(wrappers.PyEnvironmentBaseWrapper):

  def __init__(self, gym_env, observations_whitelist=None, render_kwargs=None, auxCamSettings=None):

    super(PixelObservationsGymWrapperState, self).__init__(gym_env)
    self._observations_whitelist = ['state', 'pixels', "env_info"]

    observation_spaces = collections.OrderedDict()
    for observation_name in self._observations_whitelist:
      if observation_name == 'state':
        observation_spaces['state'] = self._env.observation_space
      elif observation_name == 'pixels':
        observation_spaces['pixels'] = self._env.observation_space #### 'pixels' is actually just state itself instead of image
      elif observation_name == "env_info":
        self.max_size_of_envinfo = 5
        observation_spaces['env_info'] = gym.spaces.Box(-100*np.ones(5), 100*np.ones(self.max_size_of_envinfo))
      else:
        raise ValueError('observations_whitelist can only have "state" '
                         'or "pixels" or "env_info", got %s.' % observation_name)
    self.observation_space = gym.spaces.Dict(observation_spaces)

  def _modify_observation(self, observation, info=None):

    if info is None:
      zero_list = [0]*self.max_size_of_envinfo
      info = np.array(zero_list)

    observations = collections.OrderedDict()
    for observation_name in self._observations_whitelist:
      if observation_name == 'state':
        observations['state'] = observation
      elif observation_name == 'pixels':
        observations['pixels'] = observation
      elif observation_name == "env_info":
        observations['env_info'] = info
      else:
        raise ValueError('observations_whitelist can only have "state" '
                         'or "pixels" or "env_info", got %s.' % observation_name)
    return observations

  def _step(self, action):
    observation, reward, done, info = self._env.step(action)
    observation = self._modify_observation(observation, info)
    return observation, reward, done, info

  def _reset(self):
    observation = self._env.reset()
    return self._modify_observation(observation)

  def render(self, mode='rgb_array'):
    return self._env.render(mode=mode)